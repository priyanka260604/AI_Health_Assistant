import sys

from src.exception import CustomException


try:
    a = 10
    b = 0

    result = a / b

except Exception as e:
    raise CustomException(e, sys)

    import traceback
    traceback.print_exc()
    raise