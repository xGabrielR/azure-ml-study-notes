"""
You can run Jobs with azure cli.

    az ml job -h
    az ml job create -h

    az ml job create --file jobs_yml/hello_string.yml
                     --subscription "TODO"
                     --resource-group "TODO"
                     --workspace-name "TODO"
                     --name "Hello Job from Cli"

    az ml job create --file jobs_yml/hello_mlflow.py
                     --subscription "TODO"
                     --resource-group "TODO"
                     --workspace-name "TODO"
                     --name "Hello Mlflow from Cli"

"""