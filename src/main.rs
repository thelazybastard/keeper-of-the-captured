use std::error::Error;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    loop {
        print!("Enter Directory to clean: ");
        io::stdout().flush()?;
        let mut user_input: String = String::new();
        match io::stdin().read_line(&mut user_input) {
            Ok(bytes_read) => {
                break
            }
            Err(error) => {
                continue
            }
        } 
    }

    Ok(())
}


