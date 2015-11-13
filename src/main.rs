extern crate crossbeam;
extern crate rustc_serialize;
extern crate docopt;
extern crate image;

use docopt::Docopt;
use std::fs::File;
use std::path::Path;
use image::{GenericImage, Rgb, Pixel};

mod algos;
//use algos::{negation, canny, hough};


const USAGE : &'static str = "
Usage:
    rust_image_algos [-t | --time] <image>
    rust_image_algos (-h | --help)

Options:
    -h, --help          Show this message
    -t, --time          Print timing information
";

#[derive(Debug,RustcDecodable)]
struct Args {
    flag_time: bool,
    arg_image : String
}


fn main() {
    let args : Args = Docopt::new(USAGE)
                             .and_then(|d| d.decode() )
                             .unwrap_or_else(|e| e.exit());

    let img = image::open(&Path::new(&(args.arg_image))).unwrap_or_else(|e|{
        println!("{}", e);
        std::process::exit(1);
    });

    let (w, h) = img.dimensions();
    let rgb = img.to_rgb();
    let pixels = rgb.pixels().cloned().collect::<Vec<Rgb<u8>>>();
    let chunks : Vec<Vec<Rgb<u8>>> = pixels.chunks(w as usize).map(|p| p.iter().cloned().collect()).collect();

    algos::negation((w, h), chunks).save(&Path::new("negative.png"));




    //let mut buf : Vec<Rgb<_>> = img.pixels().map(|(_, _, p)| p.to_rgb()).collect();
    //let data : Vec<&mut [Rgb<_>]> = buf.chunks_mut(w as usize).collect();


}
