import argparse
import flet as ft
from app.app import setup_and_run
from trainers.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Badaam Sath AI Project Manager")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Start the AI training process')
    group.add_argument('--app', action='store_true', help='Launch the Flet Web UI')

    args = parser.parse_args()

    if args.app:
        setup_and_run()
    elif args.train:
        train()