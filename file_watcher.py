# file_watcher.py
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .knowledge_base import update_vector_store

logger = logging.getLogger(__name__)

class FileWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            if event.src_path.startswith(os.path.join(os.getcwd(), "docs")) or \
               event.src_path.startswith(os.path.join(os.getcwd(), "images")):
                logger.info(f"New file detected: {event.src_path}")
                update_vector_store()

def start_watcher():
    observer = Observer()
    observer.schedule(FileWatcher(), path="docs", recursive=False)
    observer.schedule(FileWatcher(), path="images", recursive=False)
    observer.start()
    logger.info("File watcher started for docs and images!")
    return observer  # Return observer to manage lifecycle if needed