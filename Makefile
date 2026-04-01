TARGET = ringvax

.PHONY: local deploy clean

local:
	streamlit run app.py

deploy: manifest.json requirements.txt
	rsconnect deploy \
		manifest manifest.json \
		--title $(TARGET)

manifest.json requirements.txt: app.py pyproject.toml uv.lock
	rm -f requirements.txt
	rsconnect write-manifest streamlit . \
		--overwrite \
		--exclude Makefile --exclude README.md

clean:
	rm -f manifest.json requirements.txt
