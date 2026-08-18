"""Generic adapters for resource models emitted by scientific agent workflows."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from systemflow.auxtypes import Message, VarCollection
from systemflow.node import Mutate, MutationInputs, MutationOutputs


@dataclass(frozen=True)
class ApplicationEstimate:
    predictions: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class _ExpressionEvaluator:
    FUNCTIONS = {
        "ceil": math.ceil,
        "floor": math.floor,
        "log": math.log,
        "log2": math.log2,
        "max": max,
        "min": min,
        "sqrt": math.sqrt,
    }

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def evaluate(self, expression: str) -> float:
        result = self._node(ast.parse(expression, mode="eval").body)
        value = float(result)
        if not math.isfinite(value):
            raise ValueError(f"Feature expression produced a non-finite value: {expression}")
        return value

    def _node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.values:
                raise ValueError(f"Unknown model input in feature expression: {node.id}")
            return self.values[node.id]
        if isinstance(node, ast.Subscript):
            value = self._node(node.value)
            index = self._node(node.slice)
            if not isinstance(index, int):
                raise ValueError("Feature subscripts must be integer constants")
            return value[index]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = self._node(node.left), self._node(node.right)
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.Pow: lambda: left**right,
                ast.Mod: lambda: left % right,
            }
            operation = operations.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported operator in feature expression")
            return operation()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = self.FUNCTIONS.get(node.func.id)
            if function is None or node.keywords:
                raise ValueError("Unsupported function in feature expression")
            return function(*(self._node(argument) for argument in node.args))
        raise ValueError(f"Unsupported syntax in feature expression: {type(node).__name__}")


class WorkflowApplicationResourceModel:
    """Evaluate a reviewed, application-independent workflow model artifact."""

    def __init__(self, model_definition: str | Path) -> None:
        path = Path(model_definition).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Workflow application model not found: {path}")
        self.path = path
        self.document = json.loads(path.read_text(encoding="utf-8"))
        if self.document.get("schema_version") != "systemflow-application-resource-model-0.1":
            raise ValueError("Unsupported workflow application resource-model schema")
        self.model_inputs = tuple(self.document.get("model_inputs", []))
        self.grouping = tuple(self.document.get("grouping", []))
        self.feature_definitions = dict(self.document.get("feature_definitions", {}))
        self.groups = list(self.document.get("groups", []))
        if not self.model_inputs or not self.feature_definitions or not self.groups:
            raise ValueError("Workflow application model is missing inputs, features, or groups")

    def _canonical_group_value(self, name: str, value: Any) -> str:
        aliases = self.document.get("group_aliases", {}).get(name, {})
        return str(aliases.get(str(value), value))

    def _select_group(self, selectors: dict[str, Any]) -> dict[str, Any]:
        if set(selectors) != set(self.grouping):
            raise ValueError(
                f"Group selectors must be exactly {sorted(self.grouping)}; "
                f"received {sorted(selectors)}"
            )
        canonical = {
            name: self._canonical_group_value(name, value)
            for name, value in selectors.items()
        }
        matches = [
            group for group in self.groups
            if all(str(group.get(name)) == canonical[name] for name in self.grouping)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one model group for {canonical}, found {len(matches)}")
        return matches[0]

    @staticmethod
    def _evaluate_target(target: dict[str, Any], features: dict[str, float]) -> float:
        value = float(target["intercept"])
        for feature, coefficient in target["standardized_coefficients"].items():
            mean = float(target["feature_means"][feature])
            scale = float(target["feature_scales"][feature])
            value += float(coefficient) * (features[feature] - mean) / scale
        return max(value, np.finfo(float).eps)

    def predict(
        self,
        inputs: dict[str, Any],
        group_selectors: dict[str, Any],
    ) -> ApplicationEstimate:
        if set(inputs) != set(self.model_inputs):
            raise ValueError(
                f"Model inputs must be exactly {sorted(self.model_inputs)}; "
                f"received {sorted(inputs)}"
            )
        features = {
            name: _ExpressionEvaluator(inputs).evaluate(expression)
            for name, expression in self.feature_definitions.items()
        }
        group = self._select_group(group_selectors)
        predictions = {
            target: self._evaluate_target(model, features)
            for target, model in group["targets"].items()
        }
        return ApplicationEstimate(
            predictions=predictions,
            metadata={
                "model_definition": str(self.path),
                "inputs": inputs,
                "group_selectors": group_selectors,
                "features": features,
            },
        )


def _transform(value: Any, transform: str) -> Any:
    if transform == "identity":
        return value
    if transform == "int":
        return int(value)
    if transform == "float":
        return float(value)
    if transform == "tuple_int":
        return tuple(int(item) for item in value)
    if transform == "list_int":
        return [int(item) for item in value]
    if transform == "str":
        return str(value)
    raise ValueError(f"Unsupported mapping transform: {transform}")


def _source_value(specification: dict[str, Any], message: Message, component: Any) -> Any:
    source = specification["source"]
    key = specification["key"]
    if source == "message.fields":
        value = message.fields[key]
    elif source == "message.properties":
        value = message.properties[key]
    elif source == "component.parameters":
        value = component.parameters[key]
    else:
        raise ValueError(f"Unsupported mapping source: {source}")
    return _transform(value, specification.get("transform", "identity"))


class ScientificApplicationModel(Mutate):
    """Generic SystemFlow mutation driven entirely by a reviewed mapping artifact."""

    def __init__(
        self,
        resource_model: WorkflowApplicationResourceModel,
        mapping: dict[str, Any],
        name: str = "Scientific application model",
    ) -> None:
        self.resource_model = resource_model
        self.mapping = mapping
        field_inputs: dict[str, str] = {}
        property_inputs: dict[str, str] = {}
        parameter_inputs: dict[str, str] = {}
        for prefix, collection in (
            ("input", mapping["model_input_mapping"]),
            ("group", mapping.get("group_mapping", {})),
        ):
            for identifier, specification in collection.items():
                attribute = f"{prefix}_{identifier}"
                source = specification["source"]
                if source == "message.fields":
                    field_inputs[attribute] = specification["key"]
                elif source == "message.properties":
                    property_inputs[attribute] = specification["key"]
                elif source == "component.parameters":
                    parameter_inputs[attribute] = specification["key"]
        outputs = mapping["output_mapping"]
        field_outputs = {
            f"target_{target}": specification["key"]
            for target, specification in outputs.items()
            if specification["destination"] == "message.fields"
        }
        property_outputs = {
            f"target_{target}": specification["key"]
            for target, specification in outputs.items()
            if specification["destination"] == "message.properties"
        }
        host_outputs = {
            f"target_{target}": specification["key"]
            for target, specification in outputs.items()
            if specification["destination"] == "host.properties"
        }
        host_outputs["model_metadata"] = mapping.get(
            "metadata_output_key", "application model metadata"
        )
        super().__init__(
            name,
            MutationInputs(
                VarCollection(**field_inputs),
                VarCollection(**property_inputs),
                VarCollection(**parameter_inputs),
            ),
            MutationOutputs(
                VarCollection(**field_outputs),
                VarCollection(**property_outputs),
                VarCollection(**host_outputs),
            ),
        )

    def transform(self, message: Message, component: Any) -> tuple[dict, dict, dict]:
        inputs = {
            name: _source_value(specification, message, component)
            for name, specification in self.mapping["model_input_mapping"].items()
        }
        selectors = {
            name: _source_value(specification, message, component)
            for name, specification in self.mapping.get("group_mapping", {}).items()
        }
        estimate = self.resource_model.predict(inputs, selectors)
        fields: dict[str, Any] = {}
        properties: dict[str, Any] = {}
        host: dict[str, Any] = {
            self.outputs.host_properties.model_metadata: estimate.metadata
        }
        for target, specification in self.mapping["output_mapping"].items():
            value = estimate.predictions[target]
            destination = specification["destination"]
            key = specification["key"]
            if destination == "message.fields":
                fields[key] = value
            elif destination == "message.properties":
                properties[key] = value
            elif destination == "host.properties":
                host[key] = value
        return fields, properties, host


class ApplicationInputSource(Mutate):
    """Small source mutation used to validate a mapped application in an ExecutionGraph."""

    def __init__(self, mapping: dict[str, Any]) -> None:
        message_specs = [
            specification
            for collection in (
                mapping["model_input_mapping"], mapping.get("group_mapping", {})
            )
            for specification in collection.values()
            if specification["source"].startswith("message.")
        ]
        parameter_inputs = {
            f"source_{index}": specification["key"]
            for index, specification in enumerate(message_specs)
        }
        field_outputs = {
            f"source_{index}": specification["key"]
            for index, specification in enumerate(message_specs)
            if specification["source"] == "message.fields"
        }
        property_outputs = {
            f"source_{index}": specification["key"]
            for index, specification in enumerate(message_specs)
            if specification["source"] == "message.properties"
        }
        self.message_specs = message_specs
        super().__init__(
            "Application input source",
            MutationInputs(VarCollection(), VarCollection(), VarCollection(**parameter_inputs)),
            MutationOutputs(
                VarCollection(**field_outputs), VarCollection(**property_outputs), VarCollection()
            ),
        )

    def transform(self, _message: Message, component: Any) -> tuple[dict, dict, dict]:
        fields: dict[str, Any] = {}
        properties: dict[str, Any] = {}
        for specification in self.message_specs:
            key = specification["key"]
            if specification["source"] == "message.fields":
                fields[key] = component.parameters[key]
            else:
                properties[key] = component.parameters[key]
        return fields, properties, {}
