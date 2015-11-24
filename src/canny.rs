use image::{Luma, Pixel, GrayImage, ImageBuffer};
use crossbeam::scope;
use super::NUM_THREADS;
use std::sync::Arc;
use simple_parallel;
use std::f64::consts::PI;
use std::iter::repeat as rpt;
use std::f64;

static CONV_MATRIX : [[f64; 7];7] = [[6.67834529e-07,2.29156256e-05,1.91165461e-04,3.87705531e-04,1.91165461e-04,2.29156256e-05,6.67834529e-07],
                   [2.29156256e-05,7.86311390e-04,6.55952325e-03,1.33034673e-02,6.55952325e-03,7.86311390e-04,2.29156256e-05],
                   [1.91165461e-04,6.55952325e-03,5.47204909e-02,1.10979447e-01,5.47204909e-02,6.55952325e-03,1.91165461e-04],
                   [3.87705531e-04,1.33034673e-02,1.10979447e-01,2.25079076e-01,1.10979447e-01,1.33034673e-02,3.87705531e-04],
                   [1.91165461e-04,6.55952325e-03,5.47204909e-02,1.10979447e-01,5.47204909e-02,6.55952325e-03,1.91165461e-04],
                   [2.29156256e-05,7.86311390e-04,6.55952325e-03,1.33034673e-02,6.55952325e-03,7.86311390e-04,2.29156256e-05],
                   [6.67834529e-07,2.29156256e-05,1.91165461e-04,3.87705531e-04,1.91165461e-04,2.29156256e-05,6.67834529e-07]];

static GX : [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
static GY : [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0;3], [1.0, 2.0, 2.0]];

const THRESH_HIGH : u8 = 200;
const THRESH_LOW  : u8 = 100;

fn convolute(dims: (u32, u32), orig: &mut ImageBuffer<Luma<u8>, Vec<u8>>) {
    let dims = (dims.0 as usize, dims.1 as usize);
    let data = Arc::new(orig.clone());
    let mut pool = simple_parallel::Pool::new(NUM_THREADS);
    let CHUNK_SIZE = dims.0 * dims.1/ NUM_THREADS;
    pool.for_(orig.chunks_mut(CHUNK_SIZE).enumerate().zip(rpt(data)), |((thread_num, chunk), data)|{
        for (i, p) in chunk.iter_mut().enumerate() {
            let mut chan = 0f64;
            let x : i64 = ((CHUNK_SIZE * thread_num + i) % dims.0 ) as i64;
            let y : i64 = ((CHUNK_SIZE * thread_num + i) / dims.1 ) as i64;
            for c in -3..4 {
                for r in -3..4 {
                    let (x, y) = (x + c, y + r);
                    if (x > 0) && (x < dims.0 as i64) && (y > 0) && (y < dims.1 as i64) {
                        let r = (r + 3) as usize;
                        let c = (c + 3) as usize;
                        let factor = CONV_MATRIX[r][c];
                        let p = data[(x as u32, y as u32)];
                        chan = chan + (p[0] as f64)*factor;
                    }
                }
            }
            *p = chan as u8;
        }
    })
}

fn round_angle(angle : f64) -> f64 {
    (angle/45.0).round() * 45.0 % 180.0
}

fn gradiant(x : f64, y : f64) -> f64 {
    (x * x + y * y).sqrt()
}

fn get_gradiants(dims : (u32, u32), blurred : &GrayImage, data : Arc<GrayImage>) -> Vec<(f64, f64)> {
    let mut pool = simple_parallel::Pool::new(NUM_THREADS);
    let dims = (dims.0 as usize , dims.1 as usize);
    let CHUNK_SIZE = dims.0 * dims.1/ NUM_THREADS;
    scope(|scope|{
        pool.map(scope, blurred.chunks(CHUNK_SIZE).enumerate().zip(rpt(data)), |((thread_num, chunk), data)|{
            chunk.into_iter().enumerate().map(|(i, p)|{
                let x : i64 = ((CHUNK_SIZE * thread_num + i) % dims.0 ) as i64;
                let y : i64 = ((CHUNK_SIZE * thread_num + i) / dims.1 ) as i64;
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

                let g = gradiant(gx,gy);
                let angle = (gy/gx).atan() + PI;
                let angle = angle * PI / 90.0;
                (g, round_angle(angle))
            }).collect::<Vec<(f64, f64)>>()
        }).flat_map(|x| x.into_iter()).collect()
    })
}

fn suppress_point(x : usize, y : usize, dims : (usize, usize), grads : &Arc<Vec<(f64, f64)>>) -> bool {
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
    (a >= strength) || (b >= strength)
}

fn suppress(dims : (u32, u32), blurred: &mut GrayImage, arc : Arc<GrayImage>, grads: Arc<Vec<(f64, f64)>>) {
    let mut pool = simple_parallel::Pool::new(NUM_THREADS);
    let dims = (dims.0 as usize, dims.1 as usize);
    let CHUNK_SIZE = dims.0 * dims.1 / NUM_THREADS;
    pool.for_(blurred.chunks_mut(CHUNK_SIZE).enumerate().zip(rpt(grads)), |((thread_num, chunk), grads)|{
        for (i, p) in chunk.iter_mut().enumerate() {
            let x = ((CHUNK_SIZE * thread_num + i) % dims.0 );
            let y = ((CHUNK_SIZE * thread_num + i) / dims.0 );
            let grad = grads[thread_num*CHUNK_SIZE + i];
            if suppress_point(x as usize, y as usize, dims, &grads) {
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

fn pixel_connected(x : i64, y : i64, arc : &Arc<GrayImage>) -> bool {
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

fn hysteresis(dims : (u32, u32), image : &mut GrayImage){
    let mut pool = simple_parallel::Pool::new(NUM_THREADS);
    let arc = Arc::new(image.clone());
    let dims = (dims.0 as usize, dims.1 as usize);
    let CHUNK_SIZE = dims.0 * dims.1 / NUM_THREADS;
    pool.for_(image.chunks_mut(CHUNK_SIZE).enumerate().zip(rpt(arc)), |((thread_num, chunk), arc)|{
        for (i, p) in chunk.iter_mut().enumerate() {
            let x : i64 = ((CHUNK_SIZE * thread_num + i) % dims.0 ) as i64;
            let y : i64 = ((CHUNK_SIZE * thread_num + i) / dims.1 ) as i64;
            if *p > THRESH_HIGH || (*p > THRESH_LOW && pixel_connected(x, y, &arc)) {
                *p = 255;
            } else {
                *p = 0;
            } 
        };
    });
}

pub fn canny(dims : (u32, u32), image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> GrayImage {
    let mut orig = image.clone();
    convolute(dims, &mut orig);
    let arc = Arc::new(orig.clone());
    let grad = get_gradiants(dims, &orig, arc.clone());
    let grad = Arc::new(grad);
    suppress(dims, &mut orig, arc.clone(), grad);
    hysteresis(dims, &mut orig);
    orig
}
