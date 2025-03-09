# mlops-list

In the world of MLOps, especially with the advent of chatbots, it's crucial to truly understand the task and the underlying technology stack. Having a solid foundation in a specific stack is essential, as it facilitates easier debugging, even when chatbots generate code for you.

Using a stack with good typing hints, such as Python with Pydantic, can improve comprehension for both humans and chatbots. The current MLOps landscape consists of LLMops and AgentOps that have emerged due to the AI hype. However, most of these new technologies are built upon the core backbone of the AI community, which relies on Langchain and Python typing.

It's essential to keep things simple, as overly complex stacks can sometimes complicate tasks that can be accomplished with a single line of code or prompt.

You'll likely spend most of your time on data preprocessing if you're handling the entire MLOps process.

For those from non-engineering backgrounds, like myself, having a solid understanding of DevOps concepts can be incredibly valuable. If you're using Windows, consider switching to Linux or a Unix-based system like Mac, as most new tools are initially compatible with these operating systems, and Windows compatibility often follows later. While Windows has a Linux subsystem, becoming comfortable with a single OS can simplify your workflow and make it easier to interact with chatbots and debug code, especially when working with research papers that often contain Linux-based code.

## Learning

First, it's necessary to understand the core concept of MLOps and what it really entails. The rest doesn't matter, including what tools to use. Remember that these are just suggestions, and in this rapidly evolving world, daily learning is necessary.

[made-with-ML](https://github.com/GokuMohandas/mlops-course)

![An Overview of what you'll learn in this course](/Docs/overview.png)
Image credits goes to [GokuMohandas](https://github.com/GokuMohandas)

[mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)

These two repositories will give you a good intuition about what to do later. Another matter to consider is whether you want to use services like AWS, Azure, or Google Cloud, or if you just want to use a Linux server. Of course, if you want to use these services, they have good instructions and documentation for their own platforms.


[deeplearniing.ai](https://learn.deeplearning.ai/)

I usually follow DeepLearning.ai by Andrew Ng. They truly introduce updated tools and their short courses are a great source for staying updated.

[awesome-mlops](https://github.com/visenger/awesome-mlops)

This awesome list is great. I use it when I'm looking for a specific tool that I couldn't find with a normal Google search or by asking ChatGPT. 

## Data prerpocessing

Data engineering and preprocessing can be considered part of today's MLOps, but it's a separate topic that deserves its own attention. However, for the purpose of this discussion, it's essential for an ML engineer or data scientist (or whatever title you prefer) to have a solid understanding of the following key concepts:

1. Data types.
2. Data formats.
3. Data conversion tools relevant to your field of work

This is because, during the debugging process, you can't completely separate the data engineering process from your work. Having a strong foundation in these areas will facilitate workflow and enable you to tackle challenges more effectively.

## Model Training and Tracking 

**Model Training**

[Pytorch](https://pytorch.org/)

Doesn't need any explalanation !

[Huggingface trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)

Hugging Face's Trainer is a high-level API designed to simplify training, fine-tuning, and evaluation of transformer models from the transformers library. It abstracts many complexities of deep learning workflows, including distributed training, mixed precision, logging, and checkpointing. Built on PyTorch and TensorFlow, Trainer supports custom datasets, metrics, and hyperparameter tuning, making it a powerful tool for NLP and beyond. By leveraging Hugging Face’s ecosystem, it integrates seamlessly with the Datasets and Accelerate libraries, enabling efficient large-scale model training with minimal code.

[Ludwig](https://github.com/ludwig-ai/ludwig)

Ludwig is a low-code framework for building custom AI models like LLMs and other deep neural networks.

**Experiment Tracking**

[Mlflow](https://github.com/mlflow/mlflow)

An open-source platform designed to streamline and manage the machine learning lifecycle, including experimentation, reproducibility, deployment, and model monitoring. It provides four key components: Tracking for logging and comparing ML experiments, Projects for packaging code in a reproducible format, Models for managing and deploying models across various platforms, and Registry for versioning and collaboration. MLflow is framework-agnostic and integrates with popular ML libraries like TensorFlow, Scikit-learn, and PyTorch, making it a versatile tool for both research and production environments.

[wandb](https://wandb.ai/)

Weights & Biases (W&B) is a powerful machine learning experiment tracking and model management tool. It provides real-time logging, visualization, and collaboration features for deep learning workflows. W&B helps researchers and engineers track model performance, compare hyperparameters, and monitor training runs seamlessly. It integrates with popular frameworks like TensorFlow, PyTorch, and Keras, making it ideal for reproducible AI research and production-level ML deployment.

**Scaling**

[Ray](https://docs.ray.io/en/latest/index.html)

Ray is an open-source framework for scaling AI and Python applications, enabling distributed computing with minimal code changes. It provides a unified interface for parallel and distributed execution, making it ideal for machine learning, data processing, and reinforcement learning tasks. Ray includes libraries like Ray Tune for hyperparameter tuning, Ray Serve for scalable model serving, and Ray RLlib for reinforcement learning. With its flexible API, it supports running workloads on a single machine or across a large cluster, optimizing resource utilization and performance in AI-driven applications.

**Data Explainability**

[Shap](https://github.com/shap/shap)

A powerful Python library for explaining the output of machine learning models. 

**Prototyping** 

[gradio](https://github.com/gradio-app/gradio) 

An open-source Python library that simplifies the process of creating and sharing user interfaces for machine learning models. With just a few lines of code, developers can build customizable web-based interfaces to demo their models, allowing users to interact with them in real-time. Gradio supports various input and output types, such as images, text, audio, and more, making it versatile for a wide range of applications. It also provides easy integration with popular machine learning frameworks like TensorFlow and PyTorch. Additionally, Gradio offers features like sharing demos via public links, embedding interfaces in notebooks, and hosting on platforms like Hugging Face Spaces, making it a powerful tool for showcasing and deploying ML models.

[streamlit](https://github.com/streamlit)

Streamlit is an open-source Python library that makes it easy to create and share custom web applications for machine learning and data science. With its simple and intuitive API, developers can quickly turn data scripts into interactive dashboards and visualizations without needing extensive web development experience. Streamlit supports a wide range of data visualization libraries, such as Matplotlib, Plotly, and Altair, and integrates seamlessly with popular data science tools like Pandas and NumPy. Its real-time updates and ease of deployment make it a popular choice for building and sharing data-driven applications. The project is actively maintained and has a growing community of contributors on GitHub.

Personal Thought: To present your prototype, you don't have to code it professionally, but you can use these two apps. I personally prefer Gradio because it is much easier, but of course, it doesn't have the flair of Streamlit!!! And you probably don't need too much color for your prototype.

[Huggingface Spaces](https://huggingface.co/spaces)

Huggingface Spaces is a platform provided by Huggingface that allows users to create, share, and explore interactive machine learning applications and demos. It enables developers and researchers to easily deploy models, datasets, and AI-powered tools in a user-friendly environment. Spaces supports a variety of frameworks, including Gradio and Streamlit, making it simple to build and showcase AI projects. Whether for prototyping, collaboration, or education, Huggingface Spaces fosters innovation and community engagement in the AI field.


## LLMops & Agentops

**LLMops (Large Language Model Operations):** LLMops is a field that focuses on the development, deployment, and management of large language models (LLMs) in various applications, such as natural language processing, text generation, and conversational AI. LLMops involves designing and optimizing the operational workflows, infrastructure, and tools required to train, deploy, and maintain LLMs.

**Agentops (Agent Operations):** Agentops is a field that deals with the design, development, and operation of autonomous agents, such as chatbots, virtual assistants, and other types of AI-powered agents. Agentops involves creating and managing the workflows, infrastructure, and tools necessary to develop, deploy, and maintain these agents, ensuring they can interact effectively with humans and other systems.

[Awesome-Prompt-Engineering](https://github.com/promptslab/Awesome-Prompt-Engineering)

To understand prompt engineering better. 

[langchain](https://github.com/langchain-ai/langchain) 

LangChain is an open-source framework designed to simplify the development of applications powered by large language models (LLMs). It provides tools and abstractions for chaining together different components, such as prompts, models, and data sources, to build sophisticated workflows. LangChain supports tasks like text generation, question answering, and summarization, while enabling integration with external APIs, databases, and other tools. Its modular design makes it highly customizable, allowing developers to create tailored solutions for natural language processing (NLP) tasks. By streamlining the process of working with LLMs, LangChain empowers developers to build intelligent, context-aware applications more efficiently.

[LlamaIndex](https://github.com/run-llama/llama_index)

LlamaIndex is a powerful framework designed to enhance the capabilities of large language models (LLMs) by enabling efficient data indexing and retrieval. It allows users to seamlessly integrate structured and unstructured data sources, making it easier to query and extract relevant information. By leveraging advanced indexing techniques, LlamaIndex improves the performance of LLMs in tasks such as question-answering, summarization, and data analysis. Its open-source nature and flexibility make it a valuable tool for developers and researchers looking to build intelligent applications with enhanced data interaction capabilities. Learn more at its 

[cewai](https://github.com/crewAIInc/crewAI)

CrewAI is an innovative framework designed to facilitate the collaboration of autonomous AI agents, enabling them to work together seamlessly on complex tasks. By leveraging the power of multi-agent systems, CrewAI allows these agents to communicate, share information, and divide responsibilities efficiently, mimicking the dynamics of a human team. This approach enhances problem-solving capabilities and scalability, making it ideal for applications ranging from automation to advanced research. With its open-source nature, CrewAI encourages community contributions and continuous improvement, fostering a collaborative ecosystem for AI development. 

[autogen](https://github.com/microsoft/autogen) 

Autogen is an open-source framework developed by Microsoft, designed to simplify the creation and management of multi-agent systems. It enables developers to build, orchestrate, and optimize workflows involving multiple autonomous agents that can collaborate to solve complex tasks. Autogen supports flexible communication patterns, dynamic agent configuration, and integration with various tools and platforms, making it a powerful solution for applications in areas like automation, AI, and distributed systems. 

[pydantic](https://docs.pydantic.dev/latest/)

Pydantic is a powerful Python library for data validation and settings management using Python type annotations. It allows developers to define data schemas with type hints, ensuring that data conforms to the specified structure and types at runtime. Pydantic is widely used for parsing and validating JSON data, environment variables, and configuration settings, making it a popular choice in web frameworks like FastAPI. 

[pydantic-ai](https://ai.pydantic.dev/) 

PydanticAI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.


## vector databases and databases
 
[postgres](https://www.postgresql.org/)

PostgreSQL, often referred to as Postgres, is a powerful, open-source relational database management system (RDBMS) known for its robustness, scalability, and compliance with SQL standards. Developed by a global community of contributors, PostgreSQL supports advanced data types, complex queries, and transactional integrity, making it a popular choice for both small and large-scale applications. It offers features like ACID compliance, foreign keys, views, triggers, and stored procedures, along with extensibility through custom functions and extensions. PostgreSQL is widely used in web, mobile, and analytics applications, and its active development community ensures continuous improvements and security updates. Its flexibility and performance make it a top choice for developers and organizations worldwide.

[faiss](https://github.com/facebookresearch/faiss)

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. 

[sqlmodel](https://github.com/fastapi/sqlmodel)

SQLModel is a library for interacting with SQL databases from Python code, with Python objects. It is designed to be intuitive, easy to use, highly compatible, and robust.

## Backend

**[taingolo](https://github.com/tiangolo?tab=overview&from=2019-12-01&to=2019-12-31) stack is great for backend :**

[fast-api](https://github.com/fastapi/fastapi)

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.

[typer](https://github.com/fastapi/typer)

 A library for building CLI applications

[ci/cd github](https://github.com/resources/articles/devops/ci-cd)

CI/CD stands for Continuous Integration and Continuous Deployment (or Continuous Delivery). It's a set of practices and tools designed to improve the software development process by automating builds, testing, and deployment, enabling you to ship code changes faster and reliably.

[git-lfs](https://github.com/git-lfs/git-lfs
)

An open source Git extension for versioning large files.

[DVC](https://github.com/iterative/dvc)

Data Versioning and ML Experiments

[docker](https://github.com/docker)
