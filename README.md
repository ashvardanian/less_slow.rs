# _Less Slow_ Rust

> The spiritual little brother of [`less_slow.cpp`](https://github.com/ashvardanian/less_slow.cpp).

Much of modern code suffers from common pitfalls: bugs, security vulnerabilities, and performance bottlenecks. University curricula often teach outdated concepts, while bootcamps oversimplify crucial software development principles.

![Less Slow Rust](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.rs.jpg?raw=true)

This repository offers practical examples of writing efficient Rust code.
The topics range from basic micro-kernels executing in a few nanoseconds to more complex constructs involving parallel algorithms, coroutines, and polymorphism.
Some of the highlights include:

- Experimental coroutines can be 3x faster than [`async-stream`](https://crates.io/crates/async-stream).
- Discouraged ["internal" math intrinsics](https://doc.rust-lang.org/std/intrinsics/) can yield a 5x speedup.

To read, jump to the `less_slow.rs` source file and read the code snippets and comments.

## Reproducing the Benchmarks

If you are familiar with Rust and want to review code and measurements as you read, you can clone the repository and execute the following commands.

```sh
rustup install nightly-2025-01-11
cargo +nightly-2025-01-11 bench
```
