use image::{ImageBuffer, GrayImage,Luma,Pixel};
use crossbeam::scope;
use super::NUM_THREADS;
use std::sync::Arc;
use std::iter::repeat as rpt; 
use simple_parallel;

pub fn hough(with_edges: &GrayImage) -> GrayImage {
    let dims = with_edges.dimensions();
    let w = dims.0 as f64;
    let h = dims.1 as f64;
    let th = ((w*w+h*h).sqrt()/2.0).round() as u32;
    let tw = 360;
    //let mut output = GrayImage::from_pixel(tw, th, Luma::from_channels(255u8, 255u8, 255u8, 255u8));
    let mut output = GrayImage::new(tw, th);
    println!("{:?}", output.dimensions());
    {
        let mut pixels = output.enumerate_pixels_mut().collect::<Vec<(u32, u32, &mut Luma<u8>)>>();
        let data = Arc::new(with_edges.clone());
        let mut pool = simple_parallel::Pool::new(NUM_THREADS);
        let CHUNK_SIZE = tw*th/NUM_THREADS as u32;
        println!("Beginning hough transform");
        pool.for_(pixels.chunks_mut(CHUNK_SIZE as usize).zip(rpt(data)), |(chunk, data)|{
            for &mut (theta,rho,ref mut p) in chunk {
                let rho = rho as f64;
                let theta = theta as f64;
                let C = theta.to_radians().cos();
                let S = theta.to_radians().sin();
                let mut totalpix = 0;
                let mut total = 0;
                if theta < 45.0 || (theta > 135.0 && theta < 225.0) || theta > 315.0 {
                    for y in 0..dims.1 {
                        let dx = w/2.0 + (rho- (h / 2.0 - y as f64)*S)/C;
                        if dx < 0.0 || dx >= w { continue }
                        let x = dx.round() as u32;
                        if x == dims.0 { continue }
                        totalpix = totalpix + 1;
                        total += data[(x, y)][0] as u64;
                    }
                } else {
                    for x in 0..dims.0 {
                        let dy = h/2.0 - (rho- (x as f64 - (w / 2.0))*C)/S;
                        if dy < 0.0 || dy >= h {continue}
                        let y = dy.round() as u32;
                        if y == dims.1 { continue }
                        totalpix = totalpix + 1;
                        total += data[(x, y)][0] as u64;
                    }
                }
                if totalpix > 0 {
                    p[0] = (total / totalpix) as u8;
                }
            }
        });
    }
    output
        //for rho in 0..th {
        ////if rho % 5 == 0 { println!("{}", rho) }
        //for theta in 0..tw {
        //let theta = theta as f64;
        //let C = theta.to_radians().cos();
        //let S = theta.to_radians().sin();
        //let mut totalpix = 0;
        //let mut total = 0;
        //if theta < 45.0 || (theta > 135.0 && theta < 225.0) || theta > 315.0 {
        //for y in 0..dims.1 {
        //let dx = (dims.0 as f64)/2.0 + (rho as f64 - (dims.1 as f64 / 2.0 - y as f64)*S)/C;
        //if dx < 0.0 || dx >= dims.0 as f64 { continue }
        //let x = dx.round() as u32;
        //if x == dims.0 { continue }
        //totalpix = totalpix + 1;
        //total += with_edges[(x, y)][0] as u64;
        //}
        //} else {
        //for x in 0..dims.0 {
        //let dy = (dims.1 as f64)/2.0 - (rho as f64 - (x as f64 - (dims.0 as f64 / 2.0)*C))/S;
        //if dy < 0.0 || dy >= dims.1 as f64 {continue}
        //let y = dy.round() as u32;
        //if y == dims.1 { continue }
        //totalpix = totalpix + 1;
        //total += with_edges[(x, y)][0] as u64;
        //}
        //}
        //if totalpix > 0 {
        //output[(rho, theta as u32)][0] = (total / totalpix) as u8;
        //}
        //}
        //}
}
