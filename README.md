# Badaam Sath AI Project Manager

This repository contains the implementation of an AI agent for the card game "Badaam Sath" (also known as Sevens or Badam Satti) using the VR-DeepPDCFR+ algorithm. It also includes a Flet-based web application for playing against the AI and scanning physical cards.

## Features

-   **VR-DeepPDCFR+ Agent:** An advanced reinforcement learning agent trained to play Badaam Sath.
-   **Training Pipeline:** A robust training system using Algorithm 1 from the paper, with support for checkpoints and metrics.
-   **Game Engine:** A complete implementation of the Badaam Sath game logic.
-   **Web UI:** A user-friendly interface built with Flet to play the game and test the bot.
-   **Card Scanner:** Integration with YOLOv8 and OpenCV to scan physical cards via webcam or image upload.

## Directory Structure

-   `agents/`: Neural network definitions and agent implementations.
-   `app/`: Flet-based web application code.
-   `checkpoints/`: Directory to store model checkpoints.
-   `game/`: Game rules and state management.
-   `memory/`: Replay buffers for training.
-   `trainers/`: Training algorithms and logic.
-   `utils/`: Helper functions and metrics.

## Installation

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install torch numpy flet opencv-python ultralytics
```

## Usage

The project is controlled via `main.py`.

### Training the Agent

To start training the AI agent:

```bash
python main.py --train
```

This will initialize the environment and start the VR-DeepPDCFR+ training loop. Checkpoints will be saved to the `checkpoints/` directory.

### Running the App

To launch the web interface:

```bash
python main.py --app
```

This will start a Flet web server (default port 8550) accessible via your browser. You can:
1.  Upload an image of your hand or use the webcam.
2.  Manually select your cards.
3.  Play against the bot and see its move suggestions.

## Requirements

-   Python 3.8+
-   CUDA-compatible GPU (recommended for training)

## Notes

-   The scanner requires a YOLO model file (e.g., `cards.pt` or `best2.pt`) to be present in the root or specified path.
-   Ensure you have the necessary `uploads` directory created by the app for file handling (it is created automatically).
