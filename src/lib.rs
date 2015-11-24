extern crate simd;
extern crate crossbeam;
extern crate image;
extern crate simple_parallel;
use image::{GrayImage, RgbImage};
pub static NUM_THREADS : usize = 8;

pub mod canny;
pub mod negation;

pub use negation::negation_simd as negation;
pub use canny::canny;
