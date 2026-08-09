import sys


class CustomException(Exception):
    """
    Custom exception class for the project.
    """

    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)

        self.error_message = self.get_detailed_error_message(
            error_message,
            error_details
        )

    @staticmethod
    def get_detailed_error_message(error_message, error_details: sys):
        """
        Returns detailed error information.
        """

        _, _, exc_tb = error_details.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"\nError occurred in Python script: [{file_name}]"
            f"\nLine Number: [{line_number}]"
            f"\nError Message: [{error_message}]"
        )

    def __str__(self):
        return self.error_message
       