# -*- coding: utf-8 -*-

"""
StopoverTools.pyt

ArcGIS Pro Python toolbox for detecting migration stopover sites from point-track telemetry data
"""

import arcpy

class Toolbox(object):
    def __init__(self):
        self.label = "Stopover Tools"
        self.alias = "stopovertools"
        self.tools = [DetectStopovers]


class DetectStopovers(object):
    def __init__(self):
        self.label = "Detect Stopover Sites"
        self.description = "Detects migration stopover sites from point-track telemetry using net displacement flat-segment detection and DBSCAN spatial clustering"

    def getParameterInfo(self):
        in_points = arcpy.Parameter(
            displayName="Input Tracking Points",
            name="in_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        in_points.filter.list = ["Point"]

        return [in_points]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessage(self, parameters):
        return

    def execute(self, parameters, message):
        in_points = parameters[0].valueAsText

        count = int(arcpy.management.GetCount(in_points)[0])
        arcpy.AddMessage("Input Layer: {}".format(in_points))
        arcpy.AddMessage("Feature Count: {}".format(count))

        desc = arcpy.Describe(in_points)
        arcpy.AddMessage("Spatial Reference: {}".format(desc.spatialReference.name))
        arcpy.AddMessage("Geometry type: {}".format(desc.shapeType))
