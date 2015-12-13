#![allow(non_snake_case)]
extern crate simd;
extern crate crossbeam;
extern crate image;
extern crate simple_parallel;
pub static NUM_THREADS : usize = 4;

pub mod canny;
pub mod negation;
pub mod hough;

pub use negation::negation_simd as negation;
pub use canny::canny;
pub use hough::hough;
