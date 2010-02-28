# -*- coding: utf-8 -*-
# from http://www.djangosnippets.org/snippets/1208/
from math import ceil
from django.core.paginator import Paginator, Page, PageNotAnInteger, EmptyPage

class CouchPaginator(Paginator):
    """Allows pagination  of couchdb-python ViewResult objects. """
    def __init__(self, object_list, per_page, orphans=0, allow_empty_first_page=True, pages_view=None):
        self.per_page = per_page
        self._object_list = object_list
        self.orphans = orphans
        self.allow_empty_first_page = allow_empty_first_page
        self._pages_view = pages_view
        self._num_pages = None

    def page(self, number):
        "Returns a Page object for the given 1-based page number."
        self._number = self.validate_number(number)
        skip = self._number * self.per_page - self.per_page
        ### FIXME it would be nice to be compatible with django
        ### and allow [from:to] syntax
        self._object_list.options['limit'] = self.per_page + 1
        self._object_list.options['skip'] = skip
        return Page(self.object_list, self._number, self)

    @property
    def _count(self):
        """Implementation specific object count (overall)"""
        if self._pages_view:
            if isinstance(self._pages_view, int):
                count = self._pages_view
            else:
                count = self._pages_view.rows[0].get("value", None)
        else:
            count = None
        return count

    @property
    def object_list(self):
        """Returns a list of results or raises EmptyPage"""
        if self._object_list:
            return list(self._object_list)[:self.per_page]
        else:
            raise EmptyPage('That page contains no results')


class SimpleCouchPaginator(CouchPaginator):
    """Allows very simple page by page pagination with CouchDB ViewResults
        use SimpleCouchPaginator only if you don't have the absolute object count.
        If you have it CouchPaginator would be fine.
    """

    def __init__(self, object_list, per_page, orphans=0, allow_empty_first_page=True):
        self.per_page = per_page
        self._object_list = object_list
        self.orphans = orphans
        self.allow_empty_first_page = allow_empty_first_page
        self._num_pages = None
        return super(SimpleCouchPaginator, self).__init__(object_list, per_page,
                        orphans=orphans, allow_empty_first_page=allow_empty_first_page)

    def validate_number(self, number):
        "Validates the given 1-based page number."
        try:
            number = int(number)
        except ValueError:
            raise PageNotAnInteger('That page number is not an integer')
        if number < 1:
            raise EmptyPage('That page number is less than 1')
        return number

    def _get_count(self):
        return False

    @property
    def num_pages(self):
        return False

    @property
    def object_count(self):
        return len(list(self._object_list))

    @property
    def next(self):
        if self.has_next:
            return self._number + 1
        else:
            return None

    @property
    def previous(self):
        if self.has_previous:
            return self._number - 1
        else:
            return None

    @property
    def has_next(self):
        return self.object_count > self.per_page

    @property
    def has_previous(self):
        return self._number > 1
