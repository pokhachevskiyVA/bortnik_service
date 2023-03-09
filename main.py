from fastapi import FastAPI, Path, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
default_text = "Управление продуктом-маркетинг\nХочу в рамках компании дорасти до директора по маркетингу, который способен давать рекомендации по определению 7P.\nИду в этом направлении. Почти сформировала команду продвижения (остался 1 человек) Обучаюсь digital-маркетингу. Команда показывает рост по приходу заявок. Сейчас хочу провести эксперимент, который может привести к увеличению продаж в конкретной категории товара. Если это получится, идею можно будет масштабировать.\nИнтересно, как работает бизнес и как работает рынок. \nХочу больше понимать клиентов, сейчас мы далеки от них, хочется быть ближе.\nТакже хочу развивать свои знания в экономике-я гуманитарий, у меня с этим безрадостно. (уверена, текст это покажет)\nХочу подтянуть знания в найме сотрудников и быстро понимать, стоит ли тратить на нгео время или нет, т.к. чутьем пользуюсь в основном. Пока не подводит. \nИнтересно сделать так, чтобы вырастить интернет продажи и вывести компанию на новый уровень дохода. Причем хочется это делать не в рамках своего отдела, а иметь возможность полностью влиять на всю цепочку(от производства продукта, запаха, цвета, качества), до скрипта продаж, логистов, цен и цвета корзины на сайте).\n\nСпасибо за тест, было интересно."
default_dialog = "Здравствуйте! Почему вы не информируете своих абонентов о повышении платы по тарифу? Не все абоненты мониторят информацию на сайте. Не все оплачивают через сайт.\n Подключение без доступа к интернету. Лицевой счёт 10123955. Расскажите пожалуйста, как происходит приостановка домашнего интернета на время отпуска/не пользования им. \nИнтересуют даты с 26.08.22 по 05.09.22. \nЛицевой счёт 10123955. Брыкова Елена Николаевна Понятно. Спасибо!\n\nА если бы возможно было, то как бы происходил перерасчёт? Спасибо, вопросов больше нет."

log_config = uvicorn.config.LOGGING_CONFIG
app = FastAPI(log_config = log_config, title="Text analytics")

@app.get("/")
def read_root():
    html_content = "<h2>Hello METANIT.COM!</h2>"
    return HTMLResponse(content=html_content)

@app.get("/texts/{text}")
async def text_statistics(text: str  = Path(min_length=2, max_length=10,\
		default='Сделал бы кривую ценности')):
    return {"len_text": len(text)}
    

@app.post("/text", response_description="Psychotyping by text",\
		 description="Calculate statisctics by text using text of quantiles")
async def text_statistics(avg_type: str = Body(min_length=7, max_length=15,\
		default='assessty_all', title='Type of quantiles',\
		description='Choose one type from \
					[\"assessty_all\", \"assessty_short\"]'),\
		text: str  = Body(min_length=5, default=default_text, \
						title='Text')):
    if avg_type not in ["assessty_all", "assessty_short"]:
    	raise HTTPException(404, "avg_type is incorrect!")
    dict_response = {"len_text": len(text), "avg_type": avg_type}
    return JSONResponse(content=dict_response, status_code=200)


@app.post("/dialogue", response_description="Psychotyping of client",\
		 description="Calculate statisctics by text using dialog-quantiles")
async def dialogue_statistics(avg_type: str = Body(min_length=7, max_length=15, default='dialogs',\
		title='Type of quantiles',\
		description='Choose one type from \
		[\"dialogs\"]'),\
		text: str  = Body(min_length=5, default=default_dialog,\
						title='Text')):
    if avg_type not in ["dialogs"]:
    	raise HTTPException(404, "avg_type is incorrect!")
    dict_response = {"len_text": len(text), "avg_type": avg_type}
    return JSONResponse(content=dict_response, status_code=200)