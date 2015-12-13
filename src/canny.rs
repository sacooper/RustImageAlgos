use image::{Luma, Pixel, GrayImage, ImageBuffer};
use crossbeam::scope;
use super::NUM_THREADS;
use std::sync::Arc;
use simple_parallel;
use std::iter::repeat as rpt; 

static GUASS_KERN : [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
static GUASS_FACTOR : f64 = 0.00390625;

static GX : [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
static GY : [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0;3], [1.0, 2.0, 1.0]];

pub const THRESH_HIGH : u8 = 120;
pub const THRESH_LOW  : u8 = 50;

mod util {
    use std::sync::Arc;
    use image::GrayImage;
    use super::THRESH_HIGH;
    use std::f64;

    pub fn round_angle(angle : f64) -> f64 {
        (angle/45.0).round() * 45.0 % 180.0
    }

    pub fn gradiant(x : f64, y : f64) -> f64 {
        (x * x + y * y).sqrt()
    }

    pub fn suppress_point(x : usize, y : usize, dims : (usize, usize), grads : &Arc<Vec<(f64, f64)>>) -> bool {
        let l = grads.len() as i64;
        let check = |x| { x >= 0 && x < l };
        let idx = y*dims.0 + x;
        let (strength, dir) = grads[idx];
        let (x, y) = (x as i64, y as i64);
        let dims = (dims.0 as i64, dims.1 as i64);
        let (idx_a, idx_b) = match dir {
            0.0 => (y*dims.0 + x-1, y*dims.0 + x+1),
            45.0 => ((y-1)*dims.0 + x+1, (y+1)*dims.1 + x-1),
            90.0 => ((y-1)*dims.0 + x, (y+1)*dims.0 + x),
            135.0 => ((y-1)*dims.0 + x-1, (y+1)*dims.0 + 1),
            _ => (-1, -1)
        };
        let a = if check(idx_a) { grads[idx_a as usize].0 } else { f64::MIN };
        let b = if check(idx_b) { grads[idx_b as usize].0 } else { f64::MIN };
        (a > strength) || (b > strength) || x < 3 || x > (dims.0 - 3) || y < 3 || y > (dims.1 - 3)
    }

    pub fn pixel_connected(x : i64, y : i64, arc : &Arc<GrayImage>) -> bool {
        let dims = arc.dimensions();
        for dx in -1..2{
            for dy in -1..2 { 
                let a = x-dx;
                let b = y-dy;
                if (a >= 0) && (a < dims.0 as i64) && (b >= 0) && (b < dims.1 as i64) && arc[(a as u32, b as u32)][0] > THRESH_HIGH {
                    return true;
                }
            }
        };
        false
    }
}

fn convolute(orig: &mut ImageBuffer<Luma<u8>, Vec<u8>>) {
    let dims = orig.dimensions();
    let dims = (dims.0 as usize, dims.1 as usize);
    let data = Arc::new(orig.clone());
    let mut pool = simple_parallel::Pool::new(NUM_THREADS);
    let CHUNK_SIZE = dims.0 * dims.1/ NUM_THREADS;
    pool.for_(orig.chunks_mut(CHUNK_SIZE).enumerate().zip(rpt(data)), |((thread_num, chunk), data)|{
        for (i, p) in chunk.iter_mut().enumerate() {
            let mut chan = 0f64;
            let x : i64 = ((CHUNK_SIZE * thread_num + i) % dims.0 ) as i64;
            let y : i64 = ((CHUNK_SIZE * thread_num + i) / dims.0 ) as i64;
            for i in -2..3 {
                let y = y+i;
                if (y >= 0) && (y < dims.1 as i64) {
                    let factor = GUASS_KERN[(i + 2) as usize];
                    let p = data[(x as u32, y as u32)];
                    chan = chan + (p[0] as f64) * factor;
                }
            }
            *p = ((chan.round() as u64) >> 4) as u8;
        }
    });
    let data = Arc::new(orig.clone());
    pool.for_(orig.chunks_mut(CHUNK_SIZE).enumerate().zip(rpt(data)), |((thread_num, chunk), data)|{
        for (i, p) in chunk.iter_mut().enumerate() {
            let mut chan = 0f64;
            let x : i64 = ((CHUNK_SIZE * thread_num + i) % dims.0 ) as i64;
            let y : i64 = ((CHUNK_SIZE * thread_num + i) / dims.0 ) as i64;
            for i in -2..3 {
                let x = x+i;
                if (x >= 0) && (x < dims.0 as i64) {
                    let factor = GUASS_KERN[(i + 2) as usize];
                    let p = data[(x as u32, y as u32)];
                    chan = chan + (((p[0] as u64) << 4) as f64) * factor;
                }
            }
            //println!("{}", (chan*GUASS_FACTOR).round());
            *p = (chan*GUASS_FACTOR).round() as u8;
        }
    });
}


fn get_gradiants(blurred : &GrayImage, data : Arc<GrayImage>) -> Vec<(f64, f64)> {
    let dims = blurred.dimensions();
    let mut pool = simple_parallel::Pool::new(NUM_THREADS);
    let dims = (dims.0 as usize , dims.1 as usize);
    let CHUNK_SIZE = dims.0 * dims.1/ NUM_THREADS;
    scope(|scope|{
        pool.map(scope, blurred.chunks(CHUNK_SIZE).enumerate().zip(rpt(data)), |((thread_num, chunk), data)|{
            chunk.into_iter().enumerate().map(|(i, _)|{
                let x : i64 = ((CHUNK_SIZE * thread_num + i) % dims.0 ) as i64;
                let y : i64 = ((CHUNK_SIZE * thread_num + i) / dims.0 ) as i64;
                let mut gx = 0f64;
                let mut gy = 0f64;
                for c in -1..2 {
                    for r in -1..2 {
                        let (x, y) = (x + c, y + r);
                        if (x > 0) && (x < dims.0 as i64) && (y > 0) && (y < dims.1 as i64) {
                            let r = (r + 1) as usize;
                            let c = (c + 1) as usize;
                            let GX_FACTOR = GX[r][c];
                            let GY_FACTOR = GY[r][c];
                            let p = data[(x as u32, y as u32)];
                            gx = gx + GX_FACTOR * p[0] as f64;
                            gy = gy + GY_FACTOR * p[0] as f64;
                        }
                    }
                }

                let g = util::gradiant(gx,gy);
                let angle = (gy).atan2(gx).to_degrees() + 180.0;
                if util::round_angle(angle).is_nan() {
                    println!("{}, {} -> {}", gy, gx, util::round_angle(angle));
                }
                (g, util::round_angle(angle))
            }).collect::<Vec<(f64, f64)>>()
        }).flat_map(|x| x.into_iter()).collect()
    })
}

fn suppress(blurred: &mut GrayImage, grads: Arc<Vec<(f64, f64)>>) {
    let dims = blurred.dimensions();
    let mut pool = simple_parallel::Pool::new(NUM_THREADS);
    let dims = (dims.0 as usize, dims.1 as usize);
    let CHUNK_SIZE = dims.0 * dims.1 / NUM_THREADS;
    pool.for_(blurred.chunks_mut(CHUNK_SIZE).enumerate().zip(rpt(grads)), |((thread_num, chunk), grads)|{
        for (i, p) in chunk.iter_mut().enumerate() {
            let x = (CHUNK_SIZE * thread_num + i) % dims.0;
            let y = (CHUNK_SIZE * thread_num + i) / dims.0;
            let grad = grads[thread_num*CHUNK_SIZE + i];
            if util::suppress_point(x as usize, y as usize, dims, &grads) {
                *p = 0;
            } else if grad.0 > 255.0 {
                *p = 255;
            } else if grad.0 < 0.0 {
                *p = 0;
            } else {
                *p = grad.0 as u8;
            }
        };
    });
}

fn hysteresis(image : &mut GrayImage){
    let dims = image.dimensions();
    let mut pool = simple_parallel::Pool::new(NUM_THREADS);
    let arc = Arc::new(image.clone());
    let dims = (dims.0 as usize, dims.1 as usize);
    let CHUNK_SIZE = dims.0 * dims.1 / NUM_THREADS;
    pool.for_(image.chunks_mut(CHUNK_SIZE).enumerate().zip(rpt(arc)), |((thread_num, chunk), arc)|{
        for (i, p) in chunk.iter_mut().enumerate() {
            let x : i64 = ((CHUNK_SIZE * thread_num + i) % dims.0 ) as i64;
            let y : i64 = ((CHUNK_SIZE * thread_num + i) / dims.0 ) as i64;
            if *p > THRESH_HIGH || (*p > THRESH_LOW && util::pixel_connected(x, y, &arc)) {
                *p = 255;
            } else {
                *p = 0;
            } 
        };
    });
}

pub fn canny(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> GrayImage {
    let mut orig = image.clone();
    convolute(&mut orig);
    let arc = Arc::new(orig.clone());
    let grad = get_gradiants(&orig, arc.clone());
    let grad = Arc::new(grad);
    suppress(&mut orig, grad);
    hysteresis(&mut orig);
    orig
}
