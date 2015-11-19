extern crate simd;
extern crate crossbeam;
extern crate image;
use simd::u8x16;
use image::{Rgb, Pixel, RgbImage, ImageBuffer};
use crossbeam::scope;

pub static NUM_THREADS : usize = 8;

mod negation {
    use simd::u8x16;
    use image::{Rgb, Pixel, RgbImage, ImageBuffer};
    use crossbeam::scope;
    use super::NUM_THREADS;

    pub fn negation_simd(dims: (u32, u32), image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> RgbImage {
        scope(|scope|{
            let mut guards = Vec::with_capacity(image.len());
            let chunk_size = ((dims.0 * dims.1) as usize) / NUM_THREADS;
            for group in image.chunks(chunk_size) {
                let guard = scope.spawn(move ||{
                    let mut v = Vec::with_capacity(group.len());
                    let mut chunks = group.chunks(16);
                    let l = chunks.len();
                    let mut temp = [0; 16];
                    for _ in 0..(l-1){
                        let t = u8x16::load(&chunks.next().unwrap(), 0);
                        (!t).store(&mut temp, 0);
                        v.extend(temp.into_iter().cloned());
                    }

                    v.extend(chunks.next().unwrap().iter().map(|&x| 255-x));
                    v

                        //chunks.enumerate().flat_map(|i, chunk|{
                        //let mut temp = [0; 16];
                        //if i == l-1 {
                        //chunk.iter().map(|&x| 255-x).collect()
                        //} else {
                        //let t = u8x16::load(&temp, 0);
                        //(!t).store(&mut temp, 0);

                        //v.extend(temp.into_iter().take(i).cloned());
                        //}
                        //})
                });
                guards.push(guard);
            }

            let vec : Vec<u8> = guards.into_iter().flat_map(|x| x.join()).collect::<Vec<u8>>();

            RgbImage::from_vec(dims.0, dims.1, vec).unwrap()
        })
    }

    pub fn negation_no_simd(dims: (u32, u32), image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> RgbImage {
        scope(|scope|{
            let mut guards = Vec::with_capacity(image.len());
            let chunk_size = ((dims.0 * dims.1) as usize) / NUM_THREADS;
            for group in image.chunks(chunk_size) {
                let guard = scope.spawn(move ||{
                    group.chunks(16).flat_map(|chunk|{
                        chunk.iter().map(|&x| 255-x).collect::<Vec<u8>>()
                    }).collect::<Vec<u8>>()
                });
                guards.push(guard);
            }

            let vec : Vec<u8> = guards.into_iter().flat_map(|x| x.join()).collect::<Vec<u8>>();

            RgbImage::from_vec(dims.0, dims.1, vec).unwrap()
        })
    }
}

mod canny {
    use simd::u8x16;
    use image::{Rgb, Pixel, RgbImage, ImageBuffer};
    use crossbeam::scope;
    use super::NUM_THREADS;
    use std::sync::Arc;

    static MULTIPLIES : f64 = 0.0062893081;     // == 1/159
    static FILTER : [[u8;5];5] = 
        [[2, 4, 5, 4, 2],
         [4, 9,12, 9, 4],
         [5,12,15,12, 5],
         [4, 9,12, 9, 4],
         [2, 4, 5, 4, 2]];

    pub fn convolute(orig: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let mut conv = orig.clone();
        scope(|scope|{
            for i in 0..NUM_THREADS {
            }
        })

    }

    pub fn canny(dims : (u32, u32), image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> RgbImage {
        image.clone()
    }
}


pub fn negation(dims: (u32, u32), image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> RgbImage {
    negation::negation_simd(dims, image)
}

pub fn canny(dims: (u32, u32), image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> RgbImage {
    canny::canny(dims, image)
}
