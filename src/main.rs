use std::error::Error;
use std::io::{self, Write};
use std::{env, fs, path};
use std::path::Path;

fn main() {

    loop {

        let home_dir: String = match env::var("HOME") {
            Ok(e) => e,
            Err(_) => return
        };

        print!("Enter Directory to clean: ");
        match io::stdout().flush() {
            Ok(_) => (),
            Err(_) => continue 
        };

        let mut user_input: String = String::new();
        match io::stdin().read_line(&mut user_input) {
            Ok(_) => (),
            Err(_) => continue         
        } 
        user_input = user_input.trim().to_string();

        let path_to_iter = Path::new(&home_dir);
        path_to_iter.join(&user_input);
        
        iterate_directory(&path_to_iter);
    }
}

fn iterate_directory(path_to_iter: &Path) {

    let check_path = Path::new(path_to_iter);
    if !check_path.exists() && !check_path.is_dir() {
        return
    }

    let image_extensions: [&str; 5] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];

    let entries = match fs::read_dir(path_to_iter) {
        Ok(e) => e,
        Err(_) => return
    };
    for entry in entries {
        
    }
}



