extern crate rustc_serialize;
extern crate docopt;
extern crate image;
extern crate time;
extern crate rust_image_algos;

use docopt::Docopt;
use std::path::Path;
use image::{GenericImage, Rgb, Pixel};

use rust_image_algos as algos;
//use algos::{negation, canny, hough};


const USAGE : &'static str = "
Usage:
    rust_image_algos [-t | --time] <image>
    rust_image_algos (-h | --help)

Options: -h, --help          Show this message
    -t, --time          Print timing information
    -d, --debug         Pring debug info
";

#[derive(Debug,RustcDecodable)]
struct Args {
    flag_time: bool,
    flag_debug: bool,
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

    let rgb = img.to_rgb();
    let grayscale = img.to_luma();

    let start = time::now();
    let neg = algos::negation(&rgb);
    let mid1 = time::now();

    let canny = algos::canny(&grayscale);

    let mid2 = time::now();
    let hough = algos::hough(&canny);
    let end = time::now();

    neg.save(&Path::new("negative.png"));
    canny.save(&Path::new("canny.png"));
    hough.save(&Path::new("hough.png"));

    let neg_dur = mid1 - start;
    let canny_dur = mid2 - mid1;
    let hough_dur = end - mid2;

    if args.flag_time {
        println!("Negation: {}us", neg_dur.num_microseconds().unwrap());
        println!("Canny:    {}us", canny_dur.num_microseconds().unwrap());
        println!("Hough:    {}us", hough_dur.num_microseconds().unwrap());
    }
}
