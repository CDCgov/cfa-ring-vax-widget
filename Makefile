TARGET = ringvax

.PHONY: local deploy clean

local:
	poetry run streamlit run app.py

deploy: manifest.json requirements.txt
	rsconnect deploy \
		manifest manifest.json \
		--title $(TARGET)

manifest.json requirements.txt: app.py pyproject.toml poetry.lock app.py
	rm -f requirements.txt
	rsconnect write-manifest streamlit . \
		--overwrite \
		--exclude Makefile --exclude README.md

clean:
	rm -f manifest.json requirements.txt
