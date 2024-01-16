### Azure Structure your repo

---

Whatever approach you take, it's best practice to agree on the standard top-level folder structure for your projects. For example, you may have the following folders in all your repos:

- .cloud: contains cloud-specific code like templates to create an Azure Machine Learning workspace.
- .ad/.github: contains Azure DevOps or GitHub artifacts like YAML pipelines to automate workflows.
- src: contains any code (Python or R scripts) used for machine learning workloads like preprocessing data or model training.
- docs: contains any Markdown files or other documentation used to describe the project.
- pipelines: contains Azure Machine Learning pipelines definitions.
- tests: contains unit and integration tests used to detect bugs and issues in your code.
- notebooks: contains Jupyter notebooks, mostly used for experimentation.

Reference:
- https://learn.microsoft.com/en-us/training/paths/introduction-machine-learn-operations/
- https://github.com/Azure/mlops-v2-ado-demo/tree/main
