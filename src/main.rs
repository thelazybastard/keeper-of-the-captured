// TODO: load ml model and feed image to ml model

use std::io::{self, Write};
use std::path::Path;
use std::{env, fs};

fn main() {
    let home_dir = match env::var("HOME") {
        Ok(e) => e,
        Err(_) => return,
    };

    print!("Enter Directory to clean: ");
    match io::stdout().flush() {
        Ok(_) => (),
        Err(_) => return,
    };

    let mut user_input = String::new();
    match io::stdin().read_line(&mut user_input) {
        Ok(_) => (),
        Err(_) => return,
    }
    user_input = user_input.trim().to_string();

    let path_to_iter = Path::new(&home_dir);
    let path_to_iter = path_to_iter.join(&user_input);
    if !path_to_iter.exists() {
        return
    };

    iterate_directory(&path_to_iter);
}

fn iterate_directory(path_to_iter: &Path) {
    let entries = match fs::read_dir(path_to_iter) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let entry_path = entry.path();

        let entry_extension = match entry_path.extension() {
            Some(e) => format!("{}", e.to_string_lossy().to_lowercase()),
            None => continue,
        };
        
        if entry_path.is_file() && ["jpg","jpeg","png","bmp","webp"].contains(&&entry_extension.as_str()) {
            println!("{}", entry_path.to_string_lossy());
        } else if entry_path.is_dir() {
            iterate_directory(&entry_path);
        }
    }
}
