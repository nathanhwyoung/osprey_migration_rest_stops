# -*- coding: utf-8 -*-

"""
StopoverTools.pyt

ArcGIS Pro Python toolbox for detecting migration stopover sites from point-track telemetry data
"""

import arcpy
import pandas as pd

class Toolbox(object):
    def __init__(self):
        self.label = "Stopover Tools"
        self.alias = "stopovertools"
        self.tools = [DetectStopovers]


class DetectStopovers(object):

    _KM_PER_UNIT = {
        "kilometers": 1.0,
        "meters": 0.001,
        "miles": 1.609344,
        "feet": 0.0003048,
        "yards": 0.0009144,
        "nauticalmiles": 1.852,
    }

    def __init__(self):
        self.label = "Detect Stopover Sites"
        self.description = "Detects migration stopover sites from point-track telemetry using net displacement flat-segment detection and DBSCAN spatial clustering"

    def getParameterInfo(self):

        # GROUP 1: INPUT AND FIELDS
        in_points = arcpy.Parameter(
            displayName="Input Tracking Points",
            name="in_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        in_points.filter.list = ["Point"]

        id_field = arcpy.Parameter(
            displayName="Individual ID Field",
            name="id_field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        id_field.parameterDependencies = [in_points.name]
        id_field.filter.list = ["Short", "Long", "Text", "GUID"]

        time_field = arcpy.Parameter(
            displayName="Timestamp Field",
            name="time_field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        time_field.parameterDependencies = [in_points.name]
        time_field.filter.list = ["Date"]

        where_clause = arcpy.Parameter(
            displayName="Expression",
            name="where_clause",
            datatype="GPSQLExpression",
            parameterType="Optional",
            direction="Input"
        )
        where_clause.parameterDependencies = [in_points.name]

        # GROUP 2: ORIGIN POINTS
        origin_method = arcpy.Parameter(
            displayName="Origin Point Source",
            name="origin_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        origin_method.filter.type = "ValueList"
        origin_method.filter.list = [
            "Derive from first fixes",
            "Use origin fields on input",
        ]
        origin_method.value = "Derive from first fixes"

        origin_days = arcpy.Parameter(
            displayName="Days of Fixes Used to Derive Origin",
            name="origin_days",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        origin_days.value = 7
        origin_days.filter.type = "Range"
        origin_days.filter.list = [1, 90]

        origin_lat_field = arcpy.Parameter(
            displayName="Origin Latitude Field",
            name="origin_lat_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input"
        )
        origin_lat_field.parameterDependencies = [in_points.name]
        origin_lat_field.filter.list = ["Float", "Single", "Double"]

        origin_lon_field = arcpy.Parameter(
            displayName="Origin Longitude Field",
            name="origin_lon_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input"
        )
        origin_lon_field.parameterDependencies = [in_points.name]
        origin_lon_field.filter.list = ["Float", "Single", "Double"]

        # GROUP 3: THRESHOLDS
        season_start = arcpy.Parameter(
            displayName="Season Start Month",
            name="season_start_month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            category="Season Filter"
        )
        season_start.filter.type = "Range"
        season_start.filter.list = [1, 12]

        season_end = arcpy.Parameter(
            displayName="Season End Month",
            name="season_end_month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            category="Season Filter"
        )
        season_end.filter.type = "Range"
        season_end.filter.list = [1, 12]

        window_days = arcpy.Parameter(
            displayName="Flat Segment Window (days)",
            name="window_days",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="Detection"
        )
        window_days.value = 7

        flat_threshold = arcpy.Parameter(
            displayName="Max Displacement Range Within Window",
            name="flat_threshold",
            datatype="GPLinearUnit",
            parameterType="Required",
            direction="Input",
            category="Detection"
        )
        flat_threshold.value = "100 Kilometers"

        min_stopover_days = arcpy.Parameter(
            displayName="Minimum Stopover Duration (days)",
            name="min_stopover_days",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="Detection"
        )
        min_stopover_days.value = 3

        min_departure = arcpy.Parameter(
            displayName="Minimum Displacement From Origin",
            name="min_departure",
            datatype="GPLinearUnit",
            parameterType="Required",
            direction="Input",
            category="Detection"
        )
        min_departure.value = "100 Kilometers"

        cluster_radius = arcpy.Parameter(
            displayName="Cluster Radius",
            name="cluster_radius",
            datatype="GPLinearUnit",
            parameterType="Required",
            direction="Input",
            category="Clustering"
        )
        cluster_radius.value = "75 Kilometers"

        min_cluster_fixes = arcpy.Parameter(
            displayName="Minimum Fixes Per Cluster",
            name="min_cluster_fixes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
            category="Clustering"
        )
        min_cluster_fixes.value = 3

        max_stopover_days = arcpy.Parameter(
            displayName="Maximum Stopover Duration (days)",
            name="max_stopover_days",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            category="Post Filters"
        )
        max_stopover_days.value = 45

        min_cluster_displacement = arcpy.Parameter(
            displayName="Minimum Cluster Displacement From Origin",
            name="min_cluster_displacement",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input",
            category="Post Filters"
        )
        min_cluster_displacement.value = "250 Kilometers"

        shared_radius = arcpy.Parameter(
            displayName="Shared Site Radius",
            name="shared_radius",
            datatype="GPLinearUnit",
            parameterType="Required",
            direction="Input",
            category="Post Filters"
        )
        shared_radius.value = "50 Kilometers"

        # GROUP 4: OUTPUTS
        out_stopovers = arcpy.Parameter(
            displayName="Output Stopover Sites",
            name="out_stopovers",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )

        out_fixes = arcpy.Parameter(
            displayName="Output Classified Fixes",
            name="out_fixes",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Output"
        )

        return [
            in_points,
            id_field,
            time_field,
            where_clause,
            origin_method,
            origin_days,
            origin_lat_field,
            origin_lon_field,
            season_start,
            season_end,
            window_days,
            flat_threshold,
            min_stopover_days,
            min_departure,
            cluster_radius,
            min_cluster_fixes,
            max_stopover_days,
            min_cluster_displacement,
            shared_radius,
            out_stopovers,
            out_fixes,
            ]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessage(self, parameters):
        return

    def execute(self, parameters, messages):
        p = self._params_by_name(parameters)

        df = self._read_points(
            p["in_points"].valueAsText,
            p["id_field"].valueAsText,
            p["time_field"].valueAsText,
            p["where_clause"].valueAsText,
        )

        arcpy.AddMessage("Fixes read         : {:,}".format(len(df)))
        arcpy.AddMessage("Individuals       : {:,}".format(df["id"].nunique()))
        arcpy.AddMessage("Date range        : {} to {}".format(
            df["timestamp"].min(), df["timestamp"].max()
        ))
        arcpy.AddMessage("Lat range         : {:.4f} to {:.4f}".format(
            df["lat"].min(), df["lat"].max()
        ))
        arcpy.AddMessage("Lon range         : {:.4f} to {:.4f}".format(
            df["lon"].min(), df["lon"].max()
        ))

    # HELPERS
    @staticmethod
    def _params_by_name(parameters):
        return {p.name: p for p in parameters}

    @staticmethod
    def _read_points(fc, id_field, time_field, where_clause):
        """
        read a point feature class into a DF with columns oid, id, timestamp, lat, lon.
        coords are always WGS84 decimal degrees regardless of the input's projection.
        """
        sr = arcpy.SpatialReference(4326)
        fields = ["OID@", id_field, time_field, "SHAPE@XY"]

        rows = []
        n_null = 0

        with arcpy.da.SearchCursor(
            fc,
            fields,
            where_clause=where_clause,
            spatial_reference = sr,
        ) as cursor:
            for oid, ind_id, ts, xy in cursor:
                if ind_id is None or ts is None or xy is None:
                    n_null += 1
                    continue
                x, y = xy
                rows.append((oid, ind_id, ts, y, x))

        df = pd.DataFrame(
            rows, columns=["oid", "id", "timestamp", "lat", "lon"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)

        df["id"] = df["id"].astype(str)

        if n_null:
            arcpy.AddWarning(
                "Skipped {:,} features with a null ID, timestamp, or geometry.".format(n_null)
            )

        return df
    
    @classmethod
    def _to_km(cls, param):
        text = param.valueAsText
        if not text:
            return None
        distance, _, units = text.partition(" ")
        key = units.replace(" ", "").lower()
        if key not in cls._KM_PER_UNIT:
            raise ValueError(
                "Unsupported distance unit: {}. Use kilometers, meters, miles, feet, yards, or nautical miles.".format(units)
            )
        return float(distance) * cls._KM_PER_UNIT[key]
