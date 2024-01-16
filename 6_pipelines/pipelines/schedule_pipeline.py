from azure.ai.ml.entities import JobSchedule
from azure.ai.ml.entities import RecurrenceTrigger

schedule_name = "iris_every_hour"

recurrence_trigger = RecurrenceTrigger(
    frequency="hour",
    interval=1,
)

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
)

job_schedule = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule
).result()