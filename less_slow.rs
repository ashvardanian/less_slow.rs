//! # Less Slow
//!
//! ## Low-level microbenchmarks for building a performance-first mindset
//!
//! This module provides microbenchmarks to build a performance-first mindset.
//! Focus is on tradeoffs of abstractions, covering costs of numerical operations,
//! micro-kernels, parallelism, and more.
//!
//! Key principles are written in Rust but apply universally, and have been reproduced
//! in other languages:
//!
//! - [C++ Benchmarks](https://github.com/ashvardanian/less_slow.cpp)
//! - [Python Benchmarks](https://github.com/ashvardanian/less_slow.py)
//! - [Go Benchmarks](https://github.com/ashvardanian/less_slow.go)
//!
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(internal_features)] // Needed for `core_intrinsics`.
#![feature(coroutines)] // Called `generator` before.
#![feature(coroutine_trait)]
#![feature(stmt_expr_attributes)]
#![feature(core_intrinsics)] // As an alternative to `-ffast-math`.

use std::cmp;
use std::iter;
use std::ops::RangeInclusive;
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// region: Numerics

// region: Accuracy vs Efficiency of Standard Libraries

//? Numerical computing is a core subject in high-performance computing (HPC)
//? research and graduate studies, yet its foundational concepts are more
//? accessible than they seem. Let's start with one of the most basic operations
//? — computing the __sine__ of a number.

/// Standard library sine implementation
#[inline(always)]
fn f64_sin(x: f64) -> f64 {
    x.sin()
}

//? Standard C library functions like `sin` and `sinf` are designed for maximum
//? accuracy, often at the cost of performance. We can explore approximations
//? to trade precision for speed.
//?
//? A common approach is using the Taylor-Maclaurin __series__ — a polynomial
//? expansion of a function around a point. By limiting the expansion to a few
//? terms, we can approximate `sin(x)` as:
//?
//?     sin(x) ≈ x - (x^3)/3! + (x^5)/5!
//?
//? This reduces the computational cost but comes with reduced accuracy.
//?
//? More on Taylor series: https://en.wikipedia.org/wiki/Taylor_series

/// Taylor-Maclaurin series approximation using `pow``
#[inline(always)]
fn f64_sin_maclaurin(x: f64) -> f64 {
    x - x.powi(3) / 6.0 + x.powi(5) / 120.0
}

/// Optimized Taylor-Maclaurin series without `pow`
#[inline(always)]
fn f64_sin_maclaurin_powless(x: f64) -> f64 {
    x - (x * x * x) / 6.0 + (x * x * x * x * x) / 120.0
}

/// Fast-math variant using target feature when available
#[inline(always)]
fn f64_sin_maclaurin_fast_math(x: f64) -> f64 {
    x - (x * x * x) / 6.0 + (x * x * x * x * x) / 120.0
}

/// Fast intrinsics variant of Taylor-Maclaurin series
#[inline(always)]
fn f64_sin_maclaurin_intrinsics(x: f64) -> f64 {
    unsafe {
        use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, fsub_fast};

        // Calculate x^3
        let x2 = fmul_fast(x, x);
        let x3 = fmul_fast(x2, x);

        // Calculate x^5
        let x4 = fmul_fast(x2, x2);
        let x5 = fmul_fast(x4, x);

        // Calculate terms
        let term1 = fdiv_fast(x3, 6.0);
        let term2 = fdiv_fast(x5, 120.0);

        // Combine terms
        let temp = fsub_fast(x, term1);
        fadd_fast(temp, term2)
    }
}

pub fn benchmark_sine(c: &mut Criterion) {
    // Generate test values in range [-2π, 2π]
    const PI: f64 = std::f64::consts::PI;
    let inputs: Vec<f64> = (0..1000)
        .map(|i| -2.0 * PI + (4.0 * PI * (i as f64) / 1000.0))
        .collect();

    fn run_sine<F>(name: &str, sine_fn: F, inputs: &[f64], c: &mut Criterion)
    where
        F: Fn(f64) -> f64,
    {
        c.bench_function(name, |b| {
            b.iter(|| {
                for &x in inputs {
                    black_box(sine_fn(black_box(x)));
                }
            })
        });
    }

    run_sine("std::sin", f64_sin, &inputs, c);
    run_sine("maclaurin", f64_sin_maclaurin, &inputs, c);
    run_sine("maclaurin_powless", f64_sin_maclaurin_powless, &inputs, c);
    run_sine(
        "maclaurin_intrinsics",
        f64_sin_maclaurin_intrinsics,
        &inputs,
        c,
    );
}

//? The results on Apple M2 Pro are:
//?   - `std::sin`: __3.1__ µs
//?   - `maclaurin`: __850__ ns
//?   - `maclaurin_powless`: __800__ ns
//?   - `maclaurin_intrinsics`: __570__ ns
//?
//? Overall, using the discouraged "fast math intrinsics" yielded a __5.4x__ speedup.

// endregion: Accuracy vs Efficiency of Standard Libraries

// endregion: Numerics

// region: Pipelines and Abstractions

//? Designing efficient micro-kernels is hard, but composing them into
//? high-level pipelines without losing performance is just as difficult.
//?
//? Consider a hypothetical numeric processing pipeline:
//?
//?   1. Generate all integers in a given range (e.g., [1, 49]).
//?   2. Filter out integers that are perfect squares.
//?   3. Expand each remaining number into its prime factors.
//?   4. Sum all prime factors from the filtered numbers.
//?
//? We'll explore four implementations of this pipeline:
//?
//?  - __Callback-based Pipeline__ using closures,
//?  - __Range-based Pipeline__ using a custom `PrimeFactors` iterator,
//?  - __Trait Objects__ with type-punning to compose stages dynamically,
//?  - __Experimental Coroutines-based Pipeline__ with `#[coroutine]` macros,
//?  - __Legacy Async Streams__ using the `async-stream` crate.

/// For demonstration, we'll replicate the pipeline on [3..=49].
const PIPE_START: u64 = 3;
const PIPE_END: u64 = 49;

/// Checks if an integer is a power of two.
#[inline(always)]
fn is_power_of_two(x: u64) -> bool {
    x.count_ones() == 1
}

/// Checks if an integer is a power of three.
#[inline(always)]
fn is_power_of_three(x: u64) -> bool {
    const MAX_POWER_OF_THREE: u64 = 12157665459056928801;
    x > 0 && (MAX_POWER_OF_THREE % x == 0)
}

// region: Closures

/// Factorizes `number` into primes, invoking `callback` for each factor.
fn prime_factors_closure<F>(mut number: u64, mut callback: F)
where
    F: FnMut(u64),
{
    let mut factor: u64 = 2;
    while number > 1 {
        if number % factor == 0 {
            callback(factor);
            number /= factor;
        } else {
            factor += if factor == 2 { 1 } else { 2 };
        }
    }
}

/// Callback-based pipeline.
fn pipeline_closures() -> (u64, u64) {
    let mut sum = 0;
    let mut count = 0;

    for value in PIPE_START..=PIPE_END {
        if !is_power_of_two(value) && !is_power_of_three(value) {
            prime_factors_closure(value, |factor| {
                sum += factor;
                count += 1;
            });
        }
    }
    (sum, count)
}

// endregion: Closures

// region: Iterators

/// Lazily evaluates the prime factors of a single integer.
struct PrimeFactors {
    number: u64,
    factor: u64,
}

impl PrimeFactors {
    /// Creates a new `PrimeFactors` iterator from `number`.
    fn new(number: u64) -> Self {
        PrimeFactors { number, factor: 2 }
    }
}

impl Iterator for PrimeFactors {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while self.number > 1 {
            if self.number % self.factor == 0 {
                self.number /= self.factor;
                return Some(self.factor);
            }
            if self.factor == 2 {
                self.factor = 3;
            } else {
                self.factor += 2;
            }
        }
        None
    }
}

/// Iterator-based pipeline.
fn pipeline_iterators() -> (u64, u64) {
    (PIPE_START..=PIPE_END)
        .filter(|&v| !is_power_of_two(v) && !is_power_of_three(v))
        .flat_map(|v| PrimeFactors::new(v)) // Use our lazy iterator
        .fold((0, 0), |(sum, count), factor| (sum + factor, count + 1))
}

// endregion: Iterators

// region: Polymorphism

/// A trait that processes a vector of data as a pipeline stage.
trait PipelineStage {
    fn process(&self, data: &mut Vec<u64>);
}

/// A stage that pushes `[start..=end]` into `data`, clearing it first.
struct ForRangeVirtual {
    start: u64,
    end: u64,
}
impl PipelineStage for ForRangeVirtual {
    fn process(&self, data: &mut Vec<u64>) {
        data.clear();
        for v in self.start..=self.end {
            data.push(v);
        }
    }
}

/// A stage that filters out elements according to a function `predicate`.
struct FilterVirtual {
    predicate: fn(u64) -> bool,
}
impl PipelineStage for FilterVirtual {
    fn process(&self, data: &mut Vec<u64>) {
        data.retain(|&x| !(self.predicate)(x));
    }
}

/// A stage that expands numbers into their prime factors, pushing them into `data`.
///
/// Uses `prime_factors_closure` to avoid building an intermediate `Vec<u64>`.
struct PrimeFactorsVirtual;
impl PipelineStage for PrimeFactorsVirtual {
    fn process(&self, data: &mut Vec<u64>) {
        let mut expanded = Vec::new();
        for &value in data.iter() {
            prime_factors_closure(value, |f| expanded.push(f));
        }
        *data = expanded;
    }
}

/// A pipeline that holds multiple stages as trait objects.
struct HomogeneousVirtualPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
}

impl HomogeneousVirtualPipeline {
    fn new() -> Self {
        Self { stages: Vec::new() }
    }
    fn add_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.stages.push(stage);
    }
}

impl PipelineStage for HomogeneousVirtualPipeline {
    fn process(&self, data: &mut Vec<u64>) {
        for stage in &self.stages {
            stage.process(data);
        }
    }
}

/// Trait-object-based pipeline.
fn pipeline_virtual() -> (u64, u64) {
    let mut pipeline = HomogeneousVirtualPipeline::new();
    pipeline.add_stage(Box::new(ForRangeVirtual {
        start: PIPE_START,
        end: PIPE_END,
    }));
    pipeline.add_stage(Box::new(FilterVirtual {
        predicate: is_power_of_two,
    }));
    pipeline.add_stage(Box::new(FilterVirtual {
        predicate: is_power_of_three,
    }));
    pipeline.add_stage(Box::new(PrimeFactorsVirtual));

    let mut data = Vec::new();
    pipeline.process(&mut data);

    let sum = data.iter().copied().sum();
    let count = data.len() as u64;
    (sum, count)
}

// endregion: Polymorphism

// region: Experimental Coroutines

/// A generator that produces integers in the range [start..=end].
fn for_range_coroutine(start: u64, end: u64) -> impl Coroutine<Yield = u64, Return = ()> {
    #[coroutine]
    move || {
        for value in start..=end {
            yield value;
        }
    }
}

/// A generator that filters values from the input generator based on a predicate.
fn filter_coroutine<G>(
    input: G,
    predicate: fn(u64) -> bool,
) -> impl Coroutine<Yield = u64, Return = ()>
where
    G: Coroutine<Yield = u64, Return = ()> + Unpin,
{
    #[coroutine]
    move || {
        let mut input = input;
        while let CoroutineState::Yielded(value) = Pin::new(&mut input).resume(()) {
            if !predicate(value) {
                yield value;
            }
        }
    }
}

/// A generator that expands values into their prime factors.
fn prime_factors_coroutine<G>(input: G) -> impl Coroutine<Yield = u64, Return = ()>
where
    G: Coroutine<Yield = u64, Return = ()> + Unpin,
{
    #[coroutine]
    move || {
        let mut input = input;
        while let CoroutineState::Yielded(mut number) = Pin::new(&mut input).resume(()) {
            let mut factor: u64 = 2;
            while number > 1 {
                if number % factor == 0 {
                    yield factor;
                    number /= factor;
                } else {
                    factor += if factor == 2 { 1 } else { 2 };
                }
            }
        }
    }
}

/// Coroutine-based pipeline.
fn pipeline_coroutines() -> (u64, u64) {
    let range_gen = for_range_coroutine(PIPE_START, PIPE_END);
    let filtered_gen = filter_coroutine(
        filter_coroutine(range_gen, is_power_of_two),
        is_power_of_three,
    );
    let factor_gen = prime_factors_coroutine(filtered_gen);

    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    let mut generator = factor_gen;

    while let CoroutineState::Yielded(factor) = Pin::new(&mut generator).resume(()) {
        sum += factor;
        count += 1;
    }

    (sum, count)
}

// endregion: Experimental Coroutines

// region: Streams or Pseudo-Coroutines

use async_stream::stream;
use futures::{Stream, StreamExt};

/// Produces a stream of integers in the range `[start, end]`.
fn for_range_stream(start: u64, end: u64) -> impl Stream<Item = u64> {
    stream! {
        for value in start..=end {
            yield value;
        }
    }
}

/// Filters a stream of integers by a given predicate.
fn filter_stream<S>(input: S, predicate: fn(u64) -> bool) -> impl Stream<Item = u64>
where
    S: Stream<Item = u64> + Unpin,
{
    input.filter(move |&value| futures::future::ready(!predicate(value)))
}

/// Lazily generates the prime factors of numbers in the input stream.
fn prime_factors_stream<S>(input: S) -> impl Stream<Item = u64>
where
    S: Stream<Item = u64> + Unpin,
{
    stream! {
        let mut input = Box::pin(input);
        while let Some(value) = input.next().await {
            let mut number = value;
            let mut factor = 2;

            while number > 1 {
                if number % factor == 0 {
                    yield factor;
                    number /= factor;
                } else {
                    factor += if factor == 2 { 1 } else { 2 };
                }
            }
        }
    }
}

/// Coroutine-based pipeline.
fn pipeline_streams() -> (u64, u64) {
    let range_stream = Box::pin(for_range_stream(PIPE_START, PIPE_END));
    let filtered_stream = Box::pin(filter_stream(
        filter_stream(range_stream, is_power_of_two),
        is_power_of_three,
    ));
    let factor_stream = prime_factors_stream(filtered_stream);

    let mut sum = 0;
    let mut count = 0;

    futures::executor::block_on(async {
        let mut factor_stream = Box::pin(factor_stream);
        while let Some(factor) = factor_stream.next().await {
            sum += factor;
            count += 1;
        }
    });

    (sum, count)
}

// endregion: Streams or Pseudo-Coroutines

pub fn benchmark_pipelines(c: &mut Criterion) {
    fn run_pipeline<F>(name: &str, pipeline_fn: F, c: &mut Criterion)
    where
        F: Fn() -> (u64, u64),
    {
        c.bench_function(name, |b| {
            b.iter(|| {
                let (sum, count) = pipeline_fn();
                black_box(sum);
                black_box(count);
            });
        });
    }

    run_pipeline("pipeline_closures", pipeline_closures, c);
    run_pipeline("pipeline_iterators", pipeline_iterators, c);
    run_pipeline("pipeline_virtual", pipeline_virtual, c);
    run_pipeline("pipeline_streams", pipeline_streams, c);
    run_pipeline("pipeline_coroutines", pipeline_coroutines, c);
}

//? The benchmarks show that the closures and iterators are the fastest,
//? with the virtual pipeline being expectedly slower due to dynamic dispatch.
//?
//?    - `pipeline_closures`: __226__ ns
//?    - `pipeline_iterators`: __229__ ns
//?    - `pipeline_virtual`: __888__ ns
//?    - `pipeline_streams`: __955__ ns
//?    - `pipeline_coroutines`: __358__ ns
//?
//? The results are comparable with C++ benchmarks, with Rust coroutines looking
//? better than C++ coroutines. Also, as expected, native experimental coroutines
//? are faster than Tokio streams - upgrade when you can!

// endregion: Pipelines and Abstractions

// region: Exceptions, Backups, Logging

// region: Logs

use chrono::{DateTime, Utc};
use std::fmt::Write as FmtWrite;
use std::panic::Location;

/// Logs a message using `format!` with embedded timestamp formatting.
fn log_format(buffer: &mut String, location: &Location, code: i32, message: &str) -> usize {
    let now: DateTime<Utc> = Utc::now();
    let formatted_time = now.format("%Y-%m-%dT%H:%M:%S%.3fZ");
    let len = buffer.len();
    write!(
        buffer,
        "{} | {}:{} <{:03}> \"{}\"\n",
        formatted_time,
        location.file(),
        location.line(),
        code,
        message
    )
    .ok();
    buffer.len() - len
}

pub fn benchmark_logs(c: &mut Criterion) {
    fn run_log<F>(name: &str, log_fn: F, c: &mut Criterion)
    where
        F: Fn(&mut String, &Location, i32, &str) -> usize,
    {
        let mut buffer = String::with_capacity(1024);
        let location = Location::caller();
        let errors = vec![
            (1, "Operation not permitted"),
            (12, "Cannot allocate memory"),
            (113, "No route to host"),
        ];
        let mut idx = 0;

        c.bench_function(name, |b| {
            b.iter(|| {
                buffer.clear();
                let (code, message) = errors[idx % 3];
                idx += 1;
                log_fn(&mut buffer, location, code, message);
                black_box(&buffer);
            });
        });
    }

    run_log("log_format", log_format, c);
}

//? Logs in Rust with `write!` and `DelayedFormat` take __394__ ns,
//? which is slower than the C-native `snprintf` and the `fmt::` library in C++.

// endregion: Logs

// endregion: Exceptions, Backups, Logging

// Group them into a benchmark suite.
criterion_group!(
    name = sine_benchmarks;
    config = Criterion::default();
    targets = benchmark_sine,
);

criterion_group!(
    name = pipelines_benchmarks;
    config = Criterion::default();
    targets = benchmark_pipelines,
);

criterion_group!(
    name = logs_benchmarks;
    config = Criterion::default();
    targets = benchmark_logs,
);

criterion_main!(sine_benchmarks, pipelines_benchmarks, logs_benchmarks);
