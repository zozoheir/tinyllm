
class ContextBuilder:

    def format_context(self, context: str) -> str:
        final_context = self.start_string + "\n" + context + "\n" + self.end_string
        return final_context