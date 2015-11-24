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

Options:
    -h, --help          Show this message
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

    let (w, h) = img.dimensions();
    let rgb = img.to_rgb();
    let neg_dur = time::Duration::span(||{
        algos::negation((w, h), &rgb).save(&Path::new("negative.png"));
    });
    
    let canny_dur = time::Duration::span(||{
        algos::canny((w, h), img.grayscale().as_luma8().unwrap()).save(&Path::new("canny.png"));
    });

    if args.flag_time {
        println!("Negation: {}ms", neg_dur.num_milliseconds());
        println!("Canny:    {}ms", canny_dur.num_milliseconds());
    }
}
