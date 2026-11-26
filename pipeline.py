import json
import time
import logging
import uuid

from pathlib import Path
from datetime import datetime, timezone

from bronze import run_bronze
from silver import run_silver
from gold import run_gold
from vector_store import run_vector_store


BASE_DIR = Path(__file__).resolve().parent

REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "pipeline.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("document_pipeline")


def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def execute_stage(stage_name, stage_function):

    start_time = time.perf_counter()

    result = {
        "stage": stage_name,
        "status": "RUNNING",
        "started_at": utc_now()
    }

    logger.info(
        "Starting stage: %s",
        stage_name
    )

    try:

        stage_report = stage_function()

        if stage_report.get("status") != "SUCCESS":

            raise RuntimeError(
                f"{stage_name} reported an unsuccessful status"
            )

        reconciliation = stage_report.get(
            "reconciliation"
        )

        if isinstance(reconciliation, dict):

            reconciliation_passed = all(
                value == "PASS"
                for value in reconciliation.values()
            )

        else:

            reconciliation_passed = (
                reconciliation == "PASS"
            )

        if not reconciliation_passed:

            raise RuntimeError(
                f"{stage_name} reconciliation failed"
            )

        result["status"] = "SUCCESS"

        result["metrics"] = stage_report

        logger.info(
            "Stage completed: %s",
            stage_name
        )

    except Exception as error:

        result["status"] = "FAILED"

        result["error"] = str(error)

        logger.exception(
            "Stage failed: %s",
            stage_name
        )

    result["completed_at"] = utc_now()

    result["execution_time_seconds"] = round(
        time.perf_counter() - start_time,
        2
    )

    return result


def run_pipeline():

    start_time = time.perf_counter()

    run_id = str(uuid.uuid4())

    pipeline_report = {
        "pipeline_name": "AI Document Data Pipeline",
        "run_id": run_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "stages": []
    }

    stages = [
        ("Bronze", run_bronze),
        ("Silver", run_silver),
        ("Gold", run_gold),
        ("Vector Store", run_vector_store)
    ]

    print("=" * 60)
    print("AI DOCUMENT DATA PIPELINE")
    print("=" * 60)

    print(f"Pipeline run ID: {run_id}")

    logger.info(
        "Pipeline started: %s",
        run_id
    )

    pipeline_failed = False

    for stage_name, stage_function in stages:

        if pipeline_failed:

            pipeline_report["stages"].append({
                "stage": stage_name,
                "status": "SKIPPED",
                "reason": "Upstream stage failed",
                "execution_time_seconds": 0
            })

            logger.warning(
                "Skipping stage: %s",
                stage_name
            )

            continue

        stage_result = execute_stage(
            stage_name,
            stage_function
        )

        pipeline_report["stages"].append(
            stage_result
        )

        if stage_result["status"] == "FAILED":

            pipeline_failed = True

    pipeline_report["status"] = (
        "FAILED"
        if pipeline_failed
        else "SUCCESS"
    )

    pipeline_report["completed_at"] = utc_now()

    pipeline_report["execution_time_seconds"] = round(
        time.perf_counter() - start_time,
        2
    )

    pipeline_report["successful_stages"] = sum(
        stage["status"] == "SUCCESS"
        for stage in pipeline_report["stages"]
    )

    pipeline_report["failed_stages"] = sum(
        stage["status"] == "FAILED"
        for stage in pipeline_report["stages"]
    )

    pipeline_report["skipped_stages"] = sum(
        stage["status"] == "SKIPPED"
        for stage in pipeline_report["stages"]
    )

    report_path = (
        REPORT_DIR / "pipeline_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pipeline_report,
            file,
            indent=4,
            default=str
        )

    history_path = (
        REPORT_DIR / "pipeline_history.jsonl"
    )

    with open(
        history_path,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                pipeline_report,
                default=str
            ) + "\n"
        )

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)

    for stage in pipeline_report["stages"]:

        print(
            f"{stage['stage']:<15} : "
            f"{stage['status']:<10} "
            f"({stage['execution_time_seconds']} seconds)"
        )

    print("-" * 60)

    print(
        f"Pipeline status   : "
        f"{pipeline_report['status']}"
    )

    print(
        f"Successful stages : "
        f"{pipeline_report['successful_stages']}"
    )

    print(
        f"Failed stages     : "
        f"{pipeline_report['failed_stages']}"
    )

    print(
        f"Skipped stages    : "
        f"{pipeline_report['skipped_stages']}"
    )

    print(
        f"Total runtime     : "
        f"{pipeline_report['execution_time_seconds']} seconds"
    )

    print(f"Report            : {report_path}")
    print(f"History           : {history_path}")
    print(f"Logs              : {LOG_DIR / 'pipeline.log'}")

    print("=" * 60)

    logger.info(
        "Pipeline finished: %s",
        pipeline_report["status"]
    )

    return pipeline_report


if __name__ == "__main__":

    report = run_pipeline()

    raise SystemExit(
        0 if report["status"] == "SUCCESS" else 1
    )