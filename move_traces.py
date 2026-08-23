import weave

# source project
src = weave.init("abrahambhatti525-santa-clara-university/soc-probe-router")
calls = list(src.get_calls(filter={"trace_roots_only": True}))
print(f"Found {len(calls)} calls")

# switch to destination
dst = weave.init("abrahambhatti525-santa-clara-university/ToolOptim")

# log each call as a new trace
for call in calls:
    with dst.call(
        name=call.op_name,
        inputs=call.inputs,
    ) as c:
        c.finish(outputs=call.output)