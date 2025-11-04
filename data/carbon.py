# Based on "Emissions Product Documentation and API Guide (003).pdf"
# Dependency requirements: Python 3.10+, polars, requests_oauth2client (documentation 🔶1-57 to 🔶1-60, 🔶1-109)
import json
import polars as pl
import requests
import tempfile
from pathlib import Path
from requests_oauth2client import OAuth2Client, OAuth2ClientCredentialsAuth  # Documentation 🔶1-111
from typing import List

# -------------------------- 1. Configure Core Parameters (replace with your credentials, documentation 🔶1-43 to 🔶1-49) --------------------------
# Get from CSIRO Data Shop order details page (Order Number: 6132) "Note(s)" tab
# IMPORTANT: Replace these with your actual credentials from CSIRO Data Shop
# Get credentials from: CSIRO Data Shop order details page (Order Number: 6132) "Note(s)" tab
# Or use environment variables: export CLIENT_ID="your_id" and export CLIENT_SECRET="your_secret"
import os
CLIENT_ID = os.getenv("CLIENT_ID", "YOUR_CLIENT_ID")  # Replace with actual client_id (e.g., documentation 🔶1-49 <UUID>)
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "YOUR_CLIENT_SECRET")  # Replace with actual client_secret (e.g., documentation 🔶1-49 <string>)

# Fixed endpoints as specified in documentation (🔶1-82: authentication endpoint; 🔶1-14, 🔶1-30: data endpoint)
AUTH_ENDPOINT = "https://login.microsoftonline.com/a815c246-a01f-4d10-bc3e-eeb6a48ef48a/oauth2/v2.0/token"
DATA_ENDPOINT = "https://senaps.eratos.com/api/sensor/v2/observations"

# Region streamid suffix as specified in documentation (🔶1-24 to 🔶1-26, 🔶1-37 to 🔶1-38)
# Optional regions: nsw (New South Wales), qld (Queensland), sa (South Australia), vic (Victoria), tas (Tasmania)
TARGET_REGIONS = ["nsw"]  # Example: only fetch New South Wales data

# Time format as specified in documentation (ISO 8601 UTC, 🔶1-18, 🔶1-33, 🔶1-67 to 🔶1-68)
START_TIME = "2018-07-01T00:00:00.000Z"  # Data start time
END_TIME = "2025-09-02T00:00:00.000Z"    # Data end time (suggest testing with 1 day of data first)

# Data output path (CSV format)
OUTPUT_PATH = Path("./data/nsw_carbon_intensity.csv")  # Output file save location
# ----------------------------------------------------------------------------------------------------------


class EmissionsDataFetcher(requests.Session):
    """
    Carbon intensity data fetching class: follows documentation "Sample Use (python)" section (🔶1-40 to 🔶1-74)
    Integrates authentication, data requests, parsing, and storage functionality
    """
    def __init__(self, client_id: str, client_secret: str) -> None:
        super().__init__()
        
        # Validate input parameters
        if not client_id or not client_secret:
            raise ValueError("client_id and client_secret cannot be empty")
        
        try:
            # Initialize OAuth2 Client Credentials authentication (documentation 🔶1-75 to 🔶1-94, 🔶1-111)
            self.oauth2_client = OAuth2Client(
                token_endpoint=AUTH_ENDPOINT,
                auth=(client_id, client_secret)
            )
            # Configure authentication scope (documentation 🔶1-84, 🔶1-87: scope must be "client_id/.default")
            self.auth = OAuth2ClientCredentialsAuth(
                client=self.oauth2_client,
                scope=f"{client_id}/.default"
            )
        except Exception as e:
            raise ValueError(f"OAuth2 client initialization failed: {e}")
        # Documentation recommended request headers (ensure JSON response format, 🔶1-72)
        self.headers = {
            "accept": "*/*",
            "content-type": "application/json"
        }
        # Data endpoint (documentation 🔶1-14, 🔶1-30)
        self.data_endpoint = DATA_ENDPOINT

    def _generate_streamid(self, regions: List[str]) -> str:
        """Generate streamid as specified in documentation (🔶1-24 to 🔶1-26, 🔶1-37 to 🔶1-38)"""
        if not regions:
            raise ValueError("Region list cannot be empty (documentation requires at least 1 NEM region to be specified)")
        return ",".join([
            f"csiro.energy.dch.agshop.regional_global_emissions.{region}"
            for region in regions
        ])

    @staticmethod
    def _parse_single_stream(data: dict, output_path: Path) -> None:
        """Parse single region data (documentation 🔶1-28 to 🔶1-36): extract timestamp and carbon intensity (gCO₂/kWh, 🔶1-39)"""
        # Get region identifier from response (documentation 🔶1-30, 🔶1-35)
        stream_id = data.get("_embedded", {}).get("stream", {}).get("_links", {}).get("self", {}).get("id")
        if not stream_id:
            raise ValueError("Single region data parsing failed: stream_id not found (documentation 🔶1-30)")
        region = stream_id.split(".")[-1]  # Extract region suffix (e.g., nsw)

        # Convert to Polars DataFrame and save as CSV format
        df = (pl.LazyFrame([
            {
                "timestamp_utc": elem.get("t"),  # Timestamp (UTC, documentation 🔶1-33)
                f"carbon_intensity_gco2_per_kwh_{region}": elem.get("v", {}).get("v")  # Carbon intensity value (documentation 🔶1-35, 🔶1-39)
            }
            for elem in data.get("results", [])  # Data stored in results field (documentation 🔶1-31)
        ])
        .with_columns([
            # Convert time format to UTC datetime (for easier subsequent analysis, documentation 🔶1-33)
            pl.col("timestamp_utc")
            .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%fZ", strict=True)
            .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
        ])
        .sort("timestamp_utc")  # Sort by time (not required by documentation, but recommended)
        .collect())  # Collect before saving
        
        # Save to specified path (CSV format)
        df.write_csv(output_path)

    @staticmethod
    def _parse_multiple_streams(data: dict, output_path: Path) -> None:
        """Parse multi-region data (documentation 🔶1-12 to 🔶1-23): support fetching multiple NEM region data simultaneously"""
        parsed_rows = []
        # Iterate through timestamps and region data in results (documentation 🔶1-16, 🔶1-21)
        for timestamp_utc, region_data in data.get("results", {}).items():
            row = {"timestamp_utc": timestamp_utc}
            # Extract carbon intensity value for each region (documentation 🔶1-22 to 🔶1-23)
            for stream_id, value_obj in region_data.items():
                region = stream_id.split(".")[-1]
                row[f"carbon_intensity_gco2_per_kwh_{region}"] = value_obj.get("v")
            parsed_rows.append(row)

        # Convert to DataFrame and save as CSV format
        df = (pl.LazyFrame(parsed_rows)
        .with_columns([
            pl.col("timestamp_utc")
            .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%fZ", strict=True)
            .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
        ])
        .sort("timestamp_utc")
        .collect())
        
        # Save to specified path (CSV format)
        df.write_csv(output_path)

    def fetch_and_save_data(self, regions: List[str], start: str, end: str, output_path: Path) -> None:
        """
        Core method: fetch data and save (documentation 🔶1-53 to 🔶1-74)
        :param regions: Target region list (e.g., ["nsw", "qld"])
        :param start: Start time (ISO 8601 UTC, documentation 🔶1-67)
        :param end: End time (ISO 8601 UTC, documentation 🔶1-68)
        :param output_path: Output file path (CSV format)
        """
        # 1. Generate streamid (documentation 🔶1-24, 🔶1-37)
        streamid = self._generate_streamid(regions)

        # 2. Send data request (streaming download to avoid excessive memory usage, documentation 🔶1-72)
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_json_path = Path(tmp_dir) / "response.json"
            with self.get(
                url=self.data_endpoint,
                params=dict(
                    streamid=streamid,  # Region identifier (documentation 🔶1-14, 🔶1-30)
                    start=start,        # Start time (documentation 🔶1-67)
                    end=end,            # End time (documentation 🔶1-68)
                    limit=99_999_999    # Maximum data amount (documentation 🔶1-14 example, ensure full period coverage)
                ),
                auth=self.auth,         # Use OAuth2 authentication
                headers=self.headers,   # Use configured request headers
                timeout=30  # Timeout (avoid failure due to network delay)
            ) as response:
                response.raise_for_status()  # If request fails (e.g., 401/404), raise error (documentation 🔶1-90)
                # Save response to temporary file (documentation 🔶1-72: streaming processing)
                with open(temp_json_path, "wb") as fp:
                    for chunk in response.iter_content(chunk_size=1024):
                        fp.write(chunk)

            # 3. Parse data (select parsing method based on number of regions, documentation 🔶1-12 to 🔶1-23, 🔶1-28 to 🔶1-36)
            with open(temp_json_path, "r") as fp:
                response_data = json.load(fp)
            
            # 4. Select parsing method based on number of regions and save data
            if len(regions) == 1:
                self._parse_single_stream(response_data, output_path)
            else:
                self._parse_multiple_streams(response_data, output_path)
        
        print(f"Data fetch successful! Saved to: {output_path.resolve()}")
        print(f"Data coverage period: {start} to {end} (UTC)")
        print(f"Target regions: {', '.join(regions)} (carbon intensity unit: gCO₂/kWh, documentation 🔶1-27, 🔶1-39)")


# -------------------------- 2. Code Execution Entry Point (documentation 🔶1-61 to 🔶1-70) --------------------------
if __name__ == "__main__":
    # Verify credentials have been replaced (avoid empty values causing authentication failure, documentation 🔶1-41, 🔶1-48)
    if CLIENT_ID == "YOUR_CLIENT_ID" or CLIENT_SECRET == "YOUR_CLIENT_SECRET":
        raise ValueError("Please replace CLIENT_ID and CLIENT_SECRET in the code first (obtain from CSIRO Data Shop order details page, documentation 🔶1-43 to 🔶1-49)")

    # Initialize data fetching client (documentation 🔶1-71)
    emissions_client = EmissionsDataFetcher(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    # Execute data fetching (configuration parameters above, documentation 🔶1-65 to 🔶1-70)
    emissions_client.fetch_and_save_data(
        regions=TARGET_REGIONS,
        start=START_TIME,
        end=END_TIME,
        output_path=OUTPUT_PATH
    )