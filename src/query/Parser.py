import re


# AST nodes for parsing query terms
class Node:
    pass


class TermNode(Node):
    def __init__(self, term):
        self.term = term


class NotNode(Node):
    def __init__(self, child):
        self.child = child


class AndNode(Node):
    def __init__(self, left, right):
        self.left, self.right = left, right


class OrNode(Node):
    def __init__(self, left, right):
        self.left, self.right = left, right


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse_phrase(self):
        term = self.consume()
        return TermNode(term)

    def parse_not(self):
        if self.peek() == "NOT":
            self.consume()
            return NotNode(self.parse_not())
        elif self.peek() == "(":
            self.consume()
            node = self.parse_or()
            if self.peek() == ")":
                self.consume()
            return node
        else:
            return self.parse_phrase()

    def parse_and(self):
        left = self.parse_not()
        while self.peek() == "AND":
            self.consume()
            right = self.parse_not()
            left = AndNode(left, right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self.parse_and()
            left = OrNode(left, right)
        return left

    def parse(self):
        return self.parse_or()
