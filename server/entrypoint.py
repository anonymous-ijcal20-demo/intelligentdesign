# encoding: utf-8

import os
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import uvicorn
from server.api.api import api_router
from server.api.routes.demo import init_model
from server.api.errors.custom import CustomException, custom_exception_handler
from server.config import CONFIG


DIR_PATH = os.path.abspath(os.path.dirname(__file__))


def get_application() -> FastAPI:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CONFIG['app']['cuda'])
    application = FastAPI(title=CONFIG['app']['name'], debug=False, version=CONFIG['app']['version'])
    application.mount("/css", StaticFiles(directory=os.path.join(DIR_PATH, 'static/css')), name="css")
    application.mount("/fonts", StaticFiles(directory=os.path.join(DIR_PATH, 'static/fonts')), name="fonts")
    application.mount("/img", StaticFiles(directory=os.path.join(DIR_PATH, 'static/img')), name="img")
    application.mount("/js", StaticFiles(directory=os.path.join(DIR_PATH, 'static/js')), name="js")
    application.mount("/mock", StaticFiles(directory=os.path.join(DIR_PATH, 'static/mock')), name="mock")
    application.mount("/three", StaticFiles(directory=os.path.join(DIR_PATH, 'static/three')), name="three")
    # application.mount("/favicon.ico", StaticFiles(directory=os.path.join(DIR_PATH, 'static/favicon.ico')), name="favicon")
    # 支持跨域
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 添加异常拦截，也可以用@app.exception_handler(CustomException)方式添加
    application.add_exception_handler(CustomException, custom_exception_handler)
    application.include_router(api_router, prefix="")
    return application


app = get_application()
init_model(os.path.join(DIR_PATH, '../data/intelligent_design/models/half.pth'))

# debug时启动方式
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=CONFIG['app']['port'])

