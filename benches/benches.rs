#![feature(test)]
extern crate test;
extern crate rust_image_algos;
extern crate image;

use test::black_box as bb;
use test::Bencher as B;
use rust_image_algos as algos;
use std::path::Path;
use image::{GenericImage};

const IMAGE: &'static str = "benches/test.png";

#[bench]
fn bench_negate_simd(b : &mut B){
    let img = image::open(&Path::new(IMAGE)).unwrap_or_else(|e|{
        println!("{}", e);
        std::process::exit(1);
    });

    let (w, h) = img.dimensions();
    let rgb = img.to_rgb();
    b.iter(||{
        bb(algos::negation::negation_simd((w, h), &rgb).save(&Path::new("negative.png")));
    })
}

#[bench]
fn bench_negate_no_simd(b : &mut B){
    let img = image::open(&Path::new(IMAGE)).unwrap_or_else(|e|{
        println!("{}", e);
        std::process::exit(1);
    });

    let (w, h) = img.dimensions();
    let rgb = img.to_rgb();
    b.iter(||{
        bb(algos::negation::negation_no_simd((w, h), &rgb).save(&Path::new("negative.png")));
    })
}
