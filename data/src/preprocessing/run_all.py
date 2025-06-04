from data_preprocessor import DataPreprocessor
import config

processor = DataPreprocessor(
    click_path=config.CLICK_PATH,
    place_path=config.PLACE_PATH,
    like_path=config.LIKE_PATH,
    user_path=config.USER_PATH
)

processor.run(output_path=config.OUTPUT_PATH)