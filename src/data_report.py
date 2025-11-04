import logging
from .data_preprocessing import load_and_prepare_data, data_quality_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    df = load_and_prepare_data()
    data_quality_report(df)


if __name__ == "__main__":
    main()


