#!/usr/bin/env python
# encoding: utf-8

from fastapi import APIRouter

from server.api.routes import demo

api_router = APIRouter()
api_router.include_router(demo.router)
