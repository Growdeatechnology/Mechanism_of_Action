# ============================================================
# uniprot_service.py
# Fetch Protein Sequence from UniProt REST API
# ============================================================

import requests


class UniProtService:

    BASE_URL = "https://rest.uniprot.org/uniprotkb/"

    @staticmethod
    def fetch_sequence(uniprot_id: str):
        """
        Fetch FASTA sequence from UniProt ID.
        Returns raw protein sequence string.
        """

        if not uniprot_id:
            raise ValueError("UniProt ID is empty.")

        url = f"{UniProtService.BASE_URL}{uniprot_id}.fasta"

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            raise ValueError("Invalid UniProt ID or UniProt service unavailable.")

        fasta = response.text

        # Remove FASTA header
        lines = fasta.split("\n")
        sequence = "".join(lines[1:]).strip()

        if not sequence:
            raise ValueError("No sequence found for this UniProt ID.")

        return sequence