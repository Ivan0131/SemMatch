from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from semmatch.utils import misc_utils
import unittest

class MiscUtilsTest(unittest.TestCase):

  # def test_camelcase_to_snakecase(self):
  #   self.assertEqual("typical_camel_case",
  #                    misc_utils.camel_to_snake("TypicalCamelCase"))
  #   self.assertEqual("numbers_fuse2gether",
  #                    misc_utils.camel_to_snake("NumbersFuse2gether"))
  #   self.assertEqual("numbers_fuse2_gether",
  #                    misc_utils.camel_to_snake("NumbersFuse2Gether"))
  #   self.assertEqual("lstm_seq2_seq",
  #                    misc_utils.camel_to_snake("LSTMSeq2Seq"))
  #   self.assertEqual("starts_lower",
  #                    misc_utils.camel_to_snake("startsLower"))
  #   self.assertEqual("starts_lower_caps",
  #                    misc_utils.camel_to_snake("startsLowerCAPS"))
  #   self.assertEqual("caps_fuse_together",
  #                    misc_utils.camel_to_snake("CapsFUSETogether"))
  #   self.assertEqual("startscap",
  #                    misc_utils.camel_to_snake("Startscap"))
  #   self.assertEqual("s_tartscap",
  #                    misc_utils.camel_to_snake("STartscap"))

  def test_snakecase_to_camelcase(self):
    self.assertEqual("TypicalCamelCase",
                     misc_utils.snake_to_camel("typical_camel_case"))
    self.assertEqual("NumbersFuse2gether",
                     misc_utils.snake_to_camel("numbers_fuse2gether"))
    self.assertEqual("NumbersFuse2Gether",
                     misc_utils.snake_to_camel("numbers_fuse2_gether"))
    self.assertEqual("LstmSeq2Seq",
                     misc_utils.snake_to_camel("lstm_seq2_seq"))


if __name__ == "__main__":
  unittest.main()
