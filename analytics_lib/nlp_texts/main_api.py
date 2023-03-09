from fastapi import FastAPI, Path, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.requests import Request
from analytics_lib.nlp_texts.API import api
import uvicorn

import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

log_config = uvicorn.config.LOGGING_CONFIG
app = FastAPI(log_config = log_config)


# @app.get("/")
# def read_root():
#     html_content = "<h2>Hello METANIT.COM!</h2>"
#     return HTMLResponse(content=html_content)

# @app.get("/texts/{text}")
# async def text_statistics(text: str  = Path(min_length=2, max_length=10)):
#     return {"len_text": len(text)}

@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    import traceback

    return Response(
        content="".join(
            traceback.format_exception(
                etype=type(exc), value=exc, tb=exc.__traceback__
            )
        )
    )

@app.post("/text_len")
async def text_statistics_avg(avg_type: str = Body(min_length=7, max_length=15, default='assessty_short'), text: str  = Body(min_length=5, default='Сделал бы кривую ценности.')):
    # dict_response = {"len_text":len(text)}
    # str_response = json.dumps(dict_response, cls=NpEncoder)
    dict_response = api(text=text, quantiles=avg_type)
    str_response = json.dumps(dict_response, cls=NpEncoder, indent=4)

    return JSONResponse(content=json.loads(str_response))


#if DISPLAY_TRACEBACK_ON_500: