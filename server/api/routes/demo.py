#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import numpy as np
import scipy.io as scio
from fastapi import APIRouter
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from dataset.mat2pic import TransFunc
from fpn.model import fpn
from server.api.schem.demo import DemoInputX2Y, DemoOutputX2Y
from utils import project_path

router = APIRouter()

MODELS: dict = {}
PREPROCESS: dict = {}
GT_HALF: dict = {}

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

TEMPLATES = Jinja2Templates(directory=os.path.join(DIR_PATH, '../../templates'))

# 提供openapi swagger 页面登陆接口


@router.post("/api/heatmap", response_model=DemoOutputX2Y)
def demo_X_Y(
        *, demo_input_X_Y:DemoInputX2Y
):
    global PREPROCESS, MODELS
    # print(demo_input_X_Y.grid)
    demo_input_X_Y.grid = list(map(lambda x: [x[1], x[0]], demo_input_X_Y.grid))

    layout_image = PREPROCESS["xyxy"](np.array(demo_input_X_Y.grid) + 1)
    layout_image = torch.from_numpy(layout_image).unsqueeze(0).unsqueeze(0).float().cuda()

    with torch.no_grad():
        preds = MODELS["rm_half"](layout_image)[0, 0, :, :].cpu().numpy()

    # session = requests.Session()
    # response = session.post('http://172.22.22.203:20081/api/fenics/heatmap',
    #                         json={
    #                             'coordinates': demo_input_X_Y.grid,
    #                             'task': 'rm_half',
    #                         })
    # heatmap = np.array(response.json()['heatmap'])

    heatmap = np.ones((50, 50)) * 298
    for point in demo_input_X_Y.grid:
        mat_index = 1 + point[0] + point[1] * 10
        heatmap += GT_HALF[mat_index]

    demo_output_X_Y = DemoOutputX2Y()
    # _, heatmap = demo_output_X_Y.new_random_result()
    # heatmap = heatmap * 10 + 298

    preds = preds * 10 + 298

    demo_output_X_Y.gen_heatmap(preds, heatmap)

    return demo_output_X_Y


@router.get("/")
def demo_root(request: Request):
    """
    前端页面路由
    :param request:
    :return:
    """
    return TEMPLATES.TemplateResponse('index.html', {"request": request})


def init_model(model_path):
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        device = torch.device('cpu')

    model = fpn(final_upsampling=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = torch.jit.trace(model, torch.Tensor(1, 1, 200, 200).cuda())
    MODELS["rm_half"] = model

    # PREPROCESS["xylist"] = TransFunc("separate", "xylist").preprocess
    # PREPROCESS["xyxy"] = TransFunc("separate", "xyxy").preprocess

    PREPROCESS.update(TransFunc("separate", "matrix").preprocess_dict)

    for mat_index in range(100):
        mat_index += 1
        GT_HALF[mat_index] = scio.loadmat(
            os.path.join(project_path, "server", "data", "half", "{}.mat".format(mat_index)))['u'] - 298
