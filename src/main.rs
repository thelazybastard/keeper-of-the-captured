use base64::{Engine as _, engine::general_purpose};
use dirs::picture_dir;
use ollama_rs::Ollama;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::{env, fs};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    images: Vec<String>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    num_gpu: i32,
    num_ctx: u32,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

#[tokio::main]
async fn main() {
    println!("Checking for required model (gemma3:4b)...");
    let check_output = Command::new("ollama")
        .arg("list")
        .output()
        .expect("Failed to run ollama list");

    let output_str = String::from_utf8_lossy(&check_output.stdout);
        if !output_str.contains("gemma3:4b") {
        println!("Model 'gemma3:4b' not found locally.");
        println!("Downloading 'gemma3:4b' (this may take a while)...");
        let pull_status = Command::new("ollama")
            .arg("pull")
            .arg("gemma3:4b")
            .status()
            .expect("Failed to execute ollama pull");
            
                if !pull_status.success() {
            println!("Failed to download model. Exiting.");
            return;
        }
        println!("Model downloaded successfully!");
        
        println!("Finalizing model on disk...");
        sleep(Duration::from_secs(3)).await;

        let client = Client::new();

        println!("Clearing model memory state...");
        let unload_req = serde_json::json!({
            "model": "gemma3:4b",
            "keep_alive": 0
        });
        let _ = client
            .post("http://localhost:11434/api/generate")
            .json(&unload_req)
            .send()
            .await;

        sleep(Duration::from_secs(2)).await;

        println!("Warming up model in GPU VRAM...");
        let warmup_req = OllamaRequest {
            model: "gemma3:4b".to_string(),
            prompt: "hi".to_string(),
            images: vec![],
            stream: false,
            options: OllamaOptions { num_gpu: 999, num_ctx: 2048 },
        };
        
        match client
            .post("http://localhost:11434/api/generate")
            .json(&warmup_req)
            .send()
            .await 
        {
            Ok(resp) => {
                let _ = resp.text().await; 
                println!("Warmup complete!");
            }
            Err(e) => {
                println!("Warmup request failed: {}", e);
            }
        }
            
    } else {
        println!("Model 'gemma3:4b' is available.");
    }

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
        println!("Path does not exist.");
        return;
    };

    iterate_directory(path_to_iter, &home_dir).await;
}

fn iterate_directory(path_to_iter: PathBuf, home_dir: &str) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
    Box::pin(async move {
        let entries = match fs::read_dir(&path_to_iter) {
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

            if entry_path.is_file() && ["jpg", "jpeg", "png", "bmp", "webp"].contains(&&entry_extension.as_str()) {
                load_ollama(&entry_path, false).await;
            } else if entry_path.is_dir() {
                iterate_directory(entry_path, &home_dir).await;
            }
        }

        loop {
            print!("Proceed with operation? (y/n): ");
            match io::stdout().flush() {
            Ok(_) => (),
            Err(_) => return,
            }
            let mut confirmation = String::new();
            match io::stdin().read_line(&mut confirmation) {
                Ok(_) => (),
                Err(_) => return,
            }
            confirmation = confirmation.trim().to_string();

            if confirmation.to_lowercase() == "y" {
                break
            } else if confirmation.to_lowercase() == "n" {
                panic!("Operation aborted!");
            } else {
                continue
            }
        }

        let entries2 = match fs::read_dir(&path_to_iter) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries2 {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let entry_path = entry.path();

            let entry_extension = match entry_path.extension() {
                Some(e) => format!("{}", e.to_string_lossy().to_lowercase()),
                None => continue,
            };

            if entry_path.is_file() && ["jpg", "jpeg", "png", "bmp", "webp"].contains(&&entry_extension.as_str()) {
                load_ollama(&entry_path, true).await;
            } else if entry_path.is_dir() {
                iterate_directory(entry_path, &home_dir).await;
            }
        }
    })
}

async fn load_ollama(entry_path: &PathBuf, operation: bool) {
    let _ollama = Ollama::default();

    let image_bytes = match fs::read(&entry_path) {
        Ok(e) => e,
        Err(_) => return,
    };

    let b64_image = general_purpose::STANDARD.encode(&image_bytes);

    let request = OllamaRequest {
        model: "gemma3:4b".to_string(),
        prompt: "Describe this image in one category. Reply to this prompt with only the exact category name.
        From this list of categories: Screenshots, Memes, Animals & Pets, Nature & Landscapes, 
        People & Portraits, Selfies, Food & Drink, Cars & Vehicles, Architecture & Buildings, 
        Technology & Gadgets, Art & Illustrations, Anime & Cartoons, Gaming, Documents & Receipts, 
        Charts & Graphs, Code & Programming, Text & Typography, Clothing & Fashion, 
        Events & Celebrations, Furniture & Interiors, Space, Sports & Fitness, Social Media, 
        Wallpapers, Misc.".to_string(),
        images: vec![b64_image],
        stream: false,
        options: OllamaOptions { 
            num_gpu: 999,
            num_ctx: 2048 
        }, 
    };

    let client = Client::new();
    let resp = match client
        .post("http://localhost:11434/api/generate")
        .json(&request)
        .send()
        .await {
            Ok(req) => match req.json::<OllamaResponse>().await {
                Ok(json) => json,
                Err(e) => {
                    println!("Failed to parse response: {}", e);
                    return;
                }
            },
            Err(e) => {
                println!("Request Failed: {}", e);
                return;
            }
        };
    if operation {
        match picture_dir() {
            Some(pictures_path) => {
                let dest_dir = pictures_path.join(resp.response.trim());
                if let Err(e) = fs::create_dir_all(&dest_dir) {
                    println!("Failed to create destination directory: {}", e);
                    return;
                }

                if let Some(file_name) = entry_path.file_name() {
                    let dest_file = dest_dir.join(file_name);
                    if let Err(e) = fs::rename(&entry_path, &dest_file) {
                        println!("Failed to move file: {}", e);
                    } else {
                        println!("Moved {:?} -> {:?}", entry_path.display(), resp.response);
                    }
                }
            }
            None => {
                println!("Could not find the Pictures directory on this system.");
            }
        }

    } else {
         println!("Image: {:?}\nDestination(in Pictures USER folder): {}\n", entry_path.display(), resp.response);
    }
}
