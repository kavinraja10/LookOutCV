import os
import logging
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Union


import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


from metrics.metrics import ImageMetricsCalculator




class CVMetrics(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    CONTRAST = auto()
    BLUR = auto()
    ORIENTATION = auto()
    BBOX_RATIO = auto()

    @property
    def requires_image(self) -> bool:
        """Check if the metric requires an image input."""
        return self in {self.CONTRAST, self.BLUR, self.ORIENTATION}


class BaseLogger:
    def __init__(
        self,
        model_name: str,
        enabled_metrics: Optional[List[CVMetrics]] = None,
        logs_dir: str = "lookout_cv_logs",
    ) -> None:
        """Initialize the logger for model monitoring.
        
        Args:
            model_name: Name of the model being monitored
            enabled_metrics: List of metrics to compute during logging
            logs_dir: Directory to store the parquet files
        """
        self.model_name = model_name
        self.enabled_metrics = enabled_metrics or []
        self.logs_dir = logs_dir

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.logs_dir, self.model_name), exist_ok=True)
        self.parquet_file = os.path.join(
            self.logs_dir,
            self.model_name,
            f"{self.model_name}_logs_{os.getpid()}.parquet",
        )

        if not os.path.exists(self.parquet_file):
            schema = self._create_schema()
            print(f"Creating new parquet file with schema: {schema}")
            empty_table = pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in schema], schema=schema
            )
            pq.write_table(empty_table, self.parquet_file)
        else:
            self._evolve_schema()

    def _create_schema(self) -> pa.schema:
        print('creating schema')
        fields = []
        for field_name in self._MANDATORY_FIELDS:
            dtype = (
                pa.string()
                if "name" in field_name or "class" in field_name
                else pa.float32()
            )
            fields.append(pa.field(field_name, dtype))
        for field in self.enabled_metrics:
            fields.append(pa.field(field.value, pa.float32()))
        print(f"Final schema fields: {[field.name for field in fields]}")
        print(f"Final schema types: {[field.type for field in fields]}")

        return pa.schema(fields)

    def _evolve_schema(self):
        print(f"Checking schema evolution for {self.parquet_file}")
        existing_table = pq.read_table(self.parquet_file)
        existing_cols = set(existing_table.schema.names)

        new_schema = self._create_schema()
        expected_cols = set(new_schema.names)

        missing_cols = expected_cols - existing_cols
        if missing_cols:
            print(f"🔄 Schema evolution: Adding columns {missing_cols}")

            for col in missing_cols:
                col_array = pa.array([None] * existing_table.num_rows, type= pa.float32())
                existing_table = existing_table.append_column(col, col_array)

                print(f"Updated schema: {existing_table.schema.names}")
                print(f"Updated schema: {existing_table.schema.types}")

            pq.write_table(existing_table, self.parquet_file)

    def calculate_image_metrics(self, image: Optional[Union[str, 'np.ndarray']] = None) -> Dict[str, Optional[float]]:
        """Calculate metrics for the given image.
        
        Args:
            image: Image data as numpy array or path to image file
            
        Returns:
            Dictionary mapping metric names to their computed values
        """
        results = {}
        if image is None or not self.enabled_metrics:
            for field in self.enabled_metrics:
                results[field.value] = None
            return results

        try:
            calc = ImageMetricsCalculator(image)
        except Exception:
            for field in self.enabled_metrics:
                results[field.value] = None
            return results

        for field in self.enabled_metrics:
            metric_name = field.value
            value = None
            method_name = f"calculate_{metric_name}"
            try:
                if hasattr(calc, method_name):
                    value = getattr(calc, method_name)()
            except Exception:
                value = None
            results[metric_name] = value

        return results

    def log_prediction(self, **kwargs: Any) -> None:
        """Log a model prediction with optional metrics.
        
        Args:
            **kwargs: Must include all mandatory fields defined in _MANDATORY_FIELDS.
                     Can include 'image' for computing image-based metrics.
        
        Raises:
            ValueError: If any mandatory field is missing
        """
        data = {}
        # Mandatory fields
        for field in self._MANDATORY_FIELDS:
            if field in kwargs:
                data[field] = kwargs[field]
            else:
                raise ValueError(f"Missing mandatory field: {field}")

        # Compute metrics only if additional fields are enabled
        computed_metrics = self.calculate_image_metrics(kwargs.get("image", None))
        data.update(computed_metrics)

        try:
            self.save_to_parquet(data)
        except Exception as e:
            raise

    def save_to_parquet(self, data: Dict[str, Any]) -> None:
        """Save data to parquet file with proper error handling.
        
        Args:
            data: Dictionary containing the values to save
            
        Raises:
            IOError: If reading/writing parquet file fails
        """
        try:
            existing_table = pq.read_table(self.parquet_file)
            schema = existing_table.schema
            row = [data.get(name, None) for name in schema.names]
        except Exception as e:
            raise IOError(f"Failed to read parquet file: {e}") from e


        arrays = []
        for i, name in enumerate(schema.names):
            field_type = schema.field(i).type
            value = row[i]

            print(f"Processing field: {name}, value: {value}, type: {field_type}")

            if value is None:
                arrays.append(pa.array([None], type=field_type))
            else:
                try:
                    if pa.types.is_integer(field_type):
                        value = int(value)
                    elif pa.types.is_floating(field_type):
                        value = float(value)
                    elif pa.types.is_boolean(field_type):
                        value = bool(value)
                    elif pa.types.is_string(field_type):
                        value = str(value)
                except Exception as e:
                    print(f"⚠️ Type conversion failed for field '{name}': {e}")
                    value = None
                arrays.append(pa.array([value], type=field_type))

        new_table = pa.Table.from_arrays(arrays, schema=schema)
        combined_table = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined_table, self.parquet_file)
