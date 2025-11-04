import logging
import time
from .train_model import main as train_main


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def retrain_once():
    logger.info("Starting retraining run...")
    train_main()
    logger.info("Retraining run complete.")


if __name__ == "__main__":
    # This script is intended to be scheduled externally (e.g., weekly via cron/Task Scheduler)
    retrain_once()


