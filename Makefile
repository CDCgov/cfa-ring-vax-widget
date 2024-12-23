ENGINE = podman
TARGET = ringvax

.PHONY: run build_container run_container

run:
	streamlit run ringvax/app.py

build_container:
	$(ENGINE) build -t $(TARGET) -f Dockerfile

run_container:
	$(ENGINE) run -p 8501:8501 --rm $(TARGET)
