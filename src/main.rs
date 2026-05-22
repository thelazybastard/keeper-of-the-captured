// TODO: load ml model and feed image to ml model

use base64::{Engine as _, engine::general_purpose};
use ollama_rs::Ollama;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::{env, fs};
use std::future::Future;
use std::pin::Pin;
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    images: Vec<String>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

struct OllamaProcessGuard(Child);

impl Drop for OllamaProcessGuard {
    fn drop(&mut self) {
        println!("\nShutting down the Ollama server...");
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::main]
async fn main() {
    println!("Starting Ollama server...");

    let child = Command::new("ollama")
        .arg("serve")
        .env("OLLAMA_GPU_LAYERS", "999")
        .env("OLLAMA_NUM_GPU", "999")
        .stdout(Stdio::null()) 
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start Ollama server. Is ollama installed and in your PATH?");
        

    let _server_guard = OllamaProcessGuard(child);

    sleep(Duration::from_secs(3)).await;

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

    iterate_directory(path_to_iter).await;
}

fn iterate_directory(path_to_iter: PathBuf) -> Pin<Box<dyn Future<Output = ()> + Send>> {
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
                load_ollama(&entry_path).await;
            } else if entry_path.is_dir() {
                iterate_directory(entry_path).await;
            }
        }
    })
}

async fn load_ollama(entry_path: &PathBuf) {
    let _ollama = Ollama::default();

    let image_bytes = match fs::read(&entry_path) {
        Ok(e) => e,
        Err(_) => return,
    };

    let b64_image = general_purpose::STANDARD.encode(&image_bytes);

    let request = OllamaRequest {
        model: "gemma3:4b".to_string(), // or whichever vision-capable tag you pulled
        prompt: "Describe this image in one category. Reply to this prompt with only the category name.
        From this list of categories: social media, 
        messaging, gaming, productivity, development, people, clothing, animals, nature, technology, 
        household, food and drink, vehicles, equipment, buildings, indoor, outdoor, celebrations, sports, 
        entertainment, work, art, photography, documents, text, colors, aesthetic, visual quality, memes, 
        screenshots, reactions, medical, scientific, maps, charts, symbols, and temporal.".to_string(),
        images: vec![b64_image],
        stream: false,
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
    
    println!("Image: {:?}\nDestination: {}\n", entry_path.display(), resp.response);
}
