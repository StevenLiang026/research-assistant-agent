"""Training-data sources other than the agent's own transcripts: the public function-calling datasets (M2).

public.py converts each public dataset's own encoding into the one sample format the trainer reads
(research_agent.train.samples); scripts/fetch_public_data.py downloads, converts, samples and writes the
manifest. Dedup, splitting and leakage audits are the owner's module and are not here.
"""
