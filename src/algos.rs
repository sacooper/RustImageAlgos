use image::{Rgb, Pixel, RgbImage};
use crossbeam::scope;

pub fn negation(dims: (u32, u32), image: Vec<Vec<Rgb<u8>>>) -> RgbImage {
    scope(|scope|{
        let vec_guard = scope.spawn(||{ Vec::with_capacity((dims.0*dims.1) as usize)});
        let mut guards = Vec::with_capacity(image.len());
        for mut row in image {
            let guard = scope.spawn(move ||{
                for p in row.iter_mut() {
                    p.apply(|x| 255 - x)
                }
                row
            });
            guards.push(guard);
        }

        let mut vec = vec_guard.join();
        for guard in guards {
            vec.extend(guard.join().iter().flat_map(|p| p.channels()))
        }

        RgbImage::from_vec(dims.0, dims.1, vec).unwrap()
    })
}
