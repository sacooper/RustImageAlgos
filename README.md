# Rust Image Algos
A small literary/executable demonstrating various data parallelism in Rust via image processing. So far this includes SIMD and thread pools (using [simple_parallel](https://crates.io/crates/simple_parallel))

This uses the Piston Image crate for image conversion/loading/saving, and implement image negation, canny edge detection, and (soon) Hough Transforms.
