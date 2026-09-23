# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Device snapshots for requests temporarily absent from an LWD batch."""


class LwdDraftCache:
    """Own draft storage independently of graph/staging tensor reuse.

    Call on the runner's input-preparation/compute stream, before the previous
    batch mapping is changed. A snapshot is allocated only on batch departure.
    Request object identity prevents reuse after a request ID is recycled.
    """

    def __init__(self):
        self.entries = {}

    def preserve(self, scheduler_output, requests, prev_map, drafts):
        cached = scheduler_output.scheduled_cached_reqs
        invalid = set(scheduler_output.finished_req_ids)
        invalid.update(scheduler_output.preempted_req_ids or ())
        invalid.update(cached.resumed_req_ids)
        invalid.update(req.req_id for req in scheduler_output.scheduled_new_reqs)
        for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
            state = requests.get(req_id)
            if state is not None and computed < state.num_computed_tokens:
                invalid.add(req_id)
        for req_id in list(self.entries):
            state, _ = self.entries[req_id]
            if req_id in invalid or requests.get(req_id) is not state:
                del self.entries[req_id]

        if drafts is None or not prev_map:
            return
        departing = [
            (req_id, index)
            for req_id, index in prev_map.items()
            if req_id not in scheduler_output.num_scheduled_tokens and req_id not in invalid and req_id in requests
        ]
        if not departing:
            return
        # A single device clone rather than one kernel per departing request.
        # Views retain this owned snapshot; graph replay cannot overwrite it.
        snapshot = drafts.detach().clone()
        for req_id, index in departing:
            self.entries[req_id] = (requests[req_id], snapshot[index])

    def take(self, req_id, state, length):
        entry = self.entries.pop(req_id, None)
        if entry is None:
            return None
        owner, row = entry
        if owner is not state or length > len(row):
            return None
        return row[:length]
