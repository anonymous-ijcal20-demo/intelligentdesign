#!/usr/bin/env python
# encoding: utf-8

from starlette.requests import Request
from starlette.responses import JSONResponse


class CustomException(Exception):
    """
    自定义异常类
    """

    def __init__(self, name: str):
        self.name = name


async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )
